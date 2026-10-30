from pathlib import Path
from backend.text_processing import text_to_dataframe, dataframe_to_text
from backend.pipeline import assess, heal
ROOT=Path(__file__).resolve().parents[1]
SAMPLES=ROOT/'sample messy datasets'/'text_samples'
def test_all_text_samples_heal_without_empty_output():
    for path in sorted(SAMPLES.glob('*.txt')):
        df=text_to_dataframe(path.read_text(encoding='utf-8'))
        a=assess(df,context='text_document')
        assert len(a['issues'])>0, path.name
        h=heal(df,context='text_document')
        assert len(h['cleaned'])>0, path.name
        assert dataframe_to_text(h['cleaned']).strip(), path.name
        assert not [i for i in h['after_issues'] if i.severity=='Error'], (path.name, h['after_issues'])
def test_geographic_and_cross_sentence_semantic_repairs():
    text='''Employee E101 | department: Finance | status: In Progress | city: Pune, state: Karnataka, country: India\nEmployee E101 | department: Finance | status: In Progress\nEmployee E101 | department: Sales | status: In Progress'''
    h=heal(text_to_dataframe(text),context='text_document')
    out=dataframe_to_text(h['cleaned'])
    assert 'state: Maharashtra' in out
    assert 'department: Sales' not in out
    assert out.count('department: Finance')==3
