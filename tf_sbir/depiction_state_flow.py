"""Transport a single CLIP image direction between two fixed text contexts."""

from dataclasses import dataclass
import math

import open_clip
import torch
import torch.nn.functional as F


SOURCE = "a drawing of an object."
TARGET = "a photo of an object."


def center(x):
    return x - x.mean(dim=-1, keepdim=True)


def unit(x):
    return F.normalize(x, dim=-1)


class TextReadout:
    """Coordinate chart for the centered terminal-token sphere of OpenAI ViT-B/32."""

    def __init__(self, model):
        weight = model.text_projection.double()
        gain = model.ln_final.weight.double()
        bias = model.ln_final.bias.double()
        self.matrix = gain[:, None] * weight
        self.inverse = torch.linalg.inv(self.matrix)
        self.bias = bias @ weight
        self.normal = unit(self.matrix.sum(dim=0))
        self.offset = center(bias / gain)
        self.width = len(gain)
        if self.offset.square().sum().item() >= self.width:
            raise ValueError("This checkpoint does not admit the positive-radius chart")

    def project(self, x):
        return unit(x - (x @ self.normal)[:, None] * self.normal)

    def encode(self, state):
        return self.project(math.sqrt(self.width) * (state.double() @ self.matrix) + self.bias)

    def decode(self, image):
        a = center(self.project(image.double()) @ self.inverse)
        a2 = a.square().sum(-1, keepdim=True)
        ac = (a * self.offset).sum(-1, keepdim=True)
        radius = (ac + (ac.square() + a2 * (self.width - self.offset.square().sum())).sqrt()) / a2
        return unit(center(radius * a - self.offset)).float()


def heads(x, count):
    return x.reshape(*x.shape[:-1], count, x.shape[-1] // count)


@dataclass
class ConditionalBlock:
    block: object
    keys: torch.Tensor
    values: torch.Tensor
    attention_radius: float
    mlp_radius: float

    def attention(self, token):
        attn = self.block.attn
        tokens = self.block.ln_1(token)
        query, key, value = F.linear(tokens, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
        q = heads(query, attn.num_heads) / math.sqrt(self.keys.shape[-1])
        own_key = heads(key, attn.num_heads)
        own_value = heads(value, attn.num_heads)
        other_logits = torch.einsum("bhd,lhd->bhl", q, self.keys)
        own_logit = (q * own_key).sum(-1, keepdim=True)
        attention = torch.cat([other_logits, own_logit], -1).softmax(-1)
        past = torch.einsum("bhl,lhd->bhd", attention[..., :-1], self.values)
        output = past + attention[..., -1:] * own_value
        return self.block.ls_1(attn.out_proj(output.flatten(1)))

    def mlp(self, token):
        return self.block.ls_2(self.block.mlp(self.block.ln_2(token)))

    def field(self, state, mode):
        radius = self.attention_radius if mode == "attention" else self.mlp_radius
        residual = self.attention(radius * state) if mode == "attention" else self.mlp(radius * state)
        change = center(residual) / radius
        return change - (state * change).sum(-1, keepdim=True) * state


class TextContext:
    def __init__(self, model, sentence):
        ids = open_clip.get_tokenizer("ViT-B-32")([sentence]).to(model.positional_embedding.device)
        eot = int(ids.argmax(dim=1).item())
        mask = model.attn_mask[:eot + 1, :eot + 1]
        state = model.token_embedding(ids[:, :eot + 1]) + model.positional_embedding[:eot + 1]
        self.blocks = []
        self.attention_error = self.residual_error = self.perturbation_error = 0.
        for block in model.transformer.resblocks:
            attn = block.attn
            prefix = block.ln_1(state[0, :-1])
            _, key, value = F.linear(prefix, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
            terminal = state[:, -1]
            entry = ConditionalBlock(block, heads(key, attn.num_heads), heads(value, attn.num_heads),
                                     center(terminal).norm().item(), 0.)
            manual_attention = entry.attention(terminal)
            native_attention = block.ls_1(block.attention(q_x=block.ln_1(state), attn_mask=mask))[:, -1]
            self.attention_error = max(self.attention_error, (manual_attention - native_attention).abs().amax().item())
            after_attention = terminal + native_attention
            entry.mlp_radius = center(after_attention).norm().item()
            self.blocks.append(entry)
            # A second input validates cached prefix keys independently of the native EOT path.
            perturbed = state.clone()
            perturbed[:, -1] = terminal.roll(7, dims=-1)
            token = perturbed[:, -1] + entry.attention(perturbed[:, -1])
            manual = token + entry.mlp(token)
            expected = block(perturbed, attn_mask=mask)[:, -1]
            self.perturbation_error = max(self.perturbation_error, (expected - manual).abs().amax().item())
            state = block(state, attn_mask=mask)
            self.residual_error = max(self.residual_error, (state[:, -1] - after_attention -
                                                           entry.mlp(after_attention)).abs().amax().item())
        self.native = unit(model.encode_text(ids).float())
        self.terminal = state[:, -1]


def integrate(state, block, mode, steps, sign):
    """Runge-Kutta 4 with an explicit unit-sphere extension of the tangent field."""
    h = sign / steps
    for _ in range(steps):
        k1 = block.field(state, mode)
        k2 = block.field(unit(state + (.5 * h) * k1), mode)
        k3 = block.field(unit(state + (.5 * h) * k2), mode)
        k4 = block.field(unit(state + h * k3), mode)
        state = unit(center(state + (h / 6.) * (k1 + 2. * k2 + 2. * k3 + k4)))
    return state


def propagate(state, context, depth, steps, reverse):
    sequence = context.blocks[-depth:]
    if reverse:
        for block in reversed(sequence):
            state = integrate(state, block, "mlp", steps, -1)
            state = integrate(state, block, "attention", steps, -1)
    else:
        for block in sequence:
            state = integrate(state, block, "attention", steps, 1)
            state = integrate(state, block, "mlp", steps, 1)
    return state


class DepictionFlow:
    def __init__(self, model):
        self.chart = TextReadout(model)
        self.source = TextContext(model, SOURCE)
        self.target = TextContext(model, TARGET)

    def projected(self, x):
        return self.chart.project(x.double()).float()

    def transform(self, x, depth, steps, reverse=False):
        start, end = (self.target, self.source) if reverse else (self.source, self.target)
        state = propagate(self.chart.decode(x), start, depth, steps, True)
        state = propagate(state, end, depth, steps, False)
        return self.chart.encode(state).float()

    def fixed_shift(self, x):
        difference = self.projected(self.target.native) - self.projected(self.source.native)
        return unit(self.projected(x) + difference)
