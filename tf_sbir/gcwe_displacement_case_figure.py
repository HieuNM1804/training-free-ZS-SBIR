"""Render local benchmark examples from the fixed case-selection and rank attribution records."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from matplotlib.font_manager import FontProperties, findfont


def run():
    root=Path('/content/sbir/results/gcwe-displacement-full')
    cases={k:v for p in sorted(root.glob('word-cases-*.json')) for k,v in json.loads(p.read_text())['cases'].items()}
    if set(cases)!={'sketchy_1','sketchy_2','tuberlin'}:
        raise ValueError('Both complete case reports are required')
    rows=[(name,case) for name,items in cases.items() for case in items]
    regular=findfont(FontProperties(family='DejaVu Sans'))
    title=ImageFont.truetype(findfont(FontProperties(family='DejaVu Sans',weight='bold')),22)
    label=ImageFont.truetype(regular,17)
    small=ImageFont.truetype(regular,15)
    out=Image.new('RGB',(1040,72+205*len(rows)),(255,255,255))
    draw=ImageDraw.Draw(out)
    for x,heading in [(25,'Retrieval change'),(358,'Query sketch'),(586,'Relevant photo'),(805,'Competing photo')]:
        draw.text((x,25),heading,fill='#24282d',font=title)
    names={'sketchy_1':'Sketchy Ext-1','sketchy_2':'Sketchy Ext-2','tuberlin':'TU-Berlin'}
    for row,(split,case) in enumerate(rows):
        top=70+row*205
        draw.line((22,top,1017,top),fill='#d9dee1')
        gain=case['direction']=='gained'
        color='#126350' if gain else '#994837'
        draw.text((25,top+17),names[split],fill='#222222',font=title)
        cutoff=200 if split=='sketchy_2' else 100
        draw.text((25,top+52),('Gained' if gain else 'Lost')+f' hit in top {cutoff}',fill=color,font=label)
        ranks=case['best_relevant_rank']
        draw.text((25,top+87),'First relevant photo',fill='#50555a',font=small)
        draw.text((25,top+112),f'Rank {ranks[0]} → {ranks[1]}',fill='#222222',font=title)
        for i,path in enumerate([case['query_path'],*case['photo_paths']]):
            x=315+i*237
            with Image.open(path) as src:
                im=ImageOps.contain(src.convert('RGB'),(213,145))
                out.paste(im,(x+(222-im.width)//2,top+6+(151-im.height)//2))
            cat=Path(path).parent.name.replace('_',' ')
            cap=cat
            if i:
                pair=case['selected_pair_ranks']
                cap+=f' · {pair[0][i-1]} → {pair[1][i-1]}'
            box=draw.textbbox((0,0),cap,font=small)
            draw.text((x+(222-(box[2]-box[0]))//2,top+166),cap,fill='#26292c',font=small)
    out.save('/content/sbir/gcwe-displacement-cases.jpg',quality=91)


if __name__=='__main__':
    run()
