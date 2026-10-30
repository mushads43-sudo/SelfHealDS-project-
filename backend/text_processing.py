import re
from collections import Counter, defaultdict
import pandas as pd
from .models import Issue, issue_category

# Conservative, explainable semantic references used only when the text itself
# supplies enough evidence to choose one interpretation over another.
SPELLING = {
    'recieve':'receive','recieved':'received','adress':'address','custmer':'customer',
    'teh':'the','seperate':'separate','occured':'occurred','definately':'definitely',
    'enviroment':'environment','managment':'management','succesful':'successful',
    'availble':'available','departmant':'department','employe':'employee',
}
STATE_CITY = {
    'mumbai':'maharashtra','pune':'maharashtra','bengaluru':'karnataka','bangalore':'karnataka',
    'chennai':'tamil nadu','hyderabad':'telangana','kolkata':'west bengal','jaipur':'rajasthan',
    'chicago':'illinois','boston':'massachusetts','austin':'texas','seattle':'washington',
}
COUNTRY_CITY = {
    'mumbai':'india','pune':'india','bengaluru':'india','bangalore':'india','chennai':'india',
    'hyderabad':'india','kolkata':'india','jaipur':'india','chicago':'united states',
    'boston':'united states','austin':'united states','seattle':'united states',
}
STATUS_CANON = {'in progress':'In Progress','in-progress':'In Progress','inprogress':'In Progress',
                'completed':'Completed','complete':'Completed','done':'Completed',
                'pending':'Pending','open':'Open','closed':'Closed'}
PLACEHOLDER_RE = re.compile(r'\b(?:N/?A|NA|unknown|null|none|missing|not available)\b', re.I)
EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
BAD_EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+\s*(?:\[at\]|\(at\)| at )\s*[A-Za-z0-9.-]+\s*(?:\[dot\]|\(dot\)| dot )\s*[A-Za-z]{2,}\b', re.I)


def text_to_dataframe(text: str) -> pd.DataFrame:
    # Preserve blank lines for faithful export while numbering every physical line.
    lines = text.replace('\r\n','\n').replace('\r','\n').split('\n')
    return pd.DataFrame({'line_no': range(1, len(lines)+1), 'text': lines})


def dataframe_to_text(df: pd.DataFrame) -> str:
    if 'text' not in df.columns:
        return ''
    return '\n'.join(df['text'].fillna('').astype(str).tolist())


def _issue(row, typ, severity, value, reason, evidence, proposed, repairable, conf, rule):
    return Issue(row=row,column='text',issue_type=typ,category=issue_category(typ),severity=severity,
                 value=value,reason=reason,evidence=evidence,proposed_repair=proposed,
                 repairable=repairable,confidence=conf,rule_id=rule,
                 confidence_basis='Deterministic textual/semantic evidence; confidence is an evidence-strength indicator, not a calibrated probability.')


def _extract_claims(lines):
    claims=[]
    patterns=[
        ('city', re.compile(r'\b(?:city|located in|location is)\s*[:=-]?\s*([A-Za-z ]{2,40})',re.I)),
        ('state', re.compile(r'\b(?:state)\s*[:=-]?\s*([A-Za-z ]{2,40})',re.I)),
        ('country', re.compile(r'\b(?:country)\s*[:=-]?\s*([A-Za-z ]{2,40})',re.I)),
        ('status', re.compile(r'\bstatus\s*[:=-]\s*([A-Za-z -]{2,30})',re.I)),
    ]
    for idx,line in enumerate(lines):
        for kind,pat in patterns:
            m=pat.search(line)
            if m:
                val=m.group(1).strip().strip('.,;')
                claims.append((idx,kind,val,m.span(1)))
    return claims


def detect_text_issues(df: pd.DataFrame):
    issues=[]
    if df.empty or 'text' not in df.columns:
        return issues
    lines=df['text'].fillna('').astype(str).tolist()

    # Duplicate non-empty sentences/lines.
    norm=[re.sub(r'\s+',' ',x.strip().lower()) for x in lines]
    first={}
    for i,n in enumerate(norm):
        if not n: continue
        if n in first:
            issues.append(_issue(i,'Duplicate text line','Warning',lines[i],
                'This non-empty line duplicates an earlier line exactly after whitespace/case normalization.',
                f'Normalized text matches line {first[n]+1}.','Remove duplicate line.',True,.99,'TXT-UNI-001'))
        else: first[n]=i

    # Spelling and whitespace.
    for i,line in enumerate(lines):
        if line != line.strip() or re.search(r' {2,}',line):
            issues.append(_issue(i,'Text whitespace inconsistency','Info',line,
                'The line contains leading/trailing or repeated whitespace.','Presentation-only inconsistency.',
                'Normalize whitespace without changing lexical content.',True,.99,'TXT-FMT-001'))
        for bad,good in SPELLING.items():
            if re.search(rf'\b{re.escape(bad)}\b',line,re.I):
                issues.append(_issue(i,'Contextual spelling error','Warning',line,
                    f"The token '{bad}' is a known misspelling in the active text-cleaning lexicon.",
                    f"Exact lexicon mapping: {bad} → {good}.",f"Replace '{bad}' with '{good}'.",True,.98,'TXT-LEX-001'))

        if PLACEHOLDER_RE.search(line):
            # Only flag placeholder when it occurs after a field-like label, not ordinary prose such as "unknown cause".
            if re.search(r'\b[A-Za-z][A-Za-z _-]{1,30}\s*[:=-]\s*(?:N/?A|NA|unknown|null|none|missing|not available)\b',line,re.I):
                issues.append(_issue(i,'Text placeholder value','Error',line,
                    'A field-like statement contains a placeholder instead of substantive content.',
                    'Label/value pattern plus exact placeholder-token match.',
                    'Infer from repeated entity context when possible; otherwise mark as unresolved.',False,.96,'TXT-MIS-001'))
        if BAD_EMAIL_RE.search(line):
            issues.append(_issue(i,'Obfuscated email format','Warning',line,
                'The line contains an email written with textual at/dot separators.',
                'A deterministic address pattern can be reconstructed without guessing characters.',
                'Normalize [at]/(at)/at and [dot]/(dot)/dot into @ and .',True,.98,'TXT-EMAIL-001'))

    claims=_extract_claims(lines)
    # Cross-field semantic city/state/country contradictions.
    by_line=defaultdict(dict)
    for idx,kind,val,span in claims: by_line[idx][kind]=(val,span)
    for idx,d in by_line.items():
        city=d.get('city',(None,None))[0]
        state=d.get('state',(None,None))[0]
        country=d.get('country',(None,None))[0]
        if city:
            c=city.lower()
            expected_state=STATE_CITY.get(c); expected_country=COUNTRY_CITY.get(c)
            if state and expected_state and state.lower()!=expected_state:
                issues.append(_issue(idx,'Text geographic contradiction','Error',lines[idx],
                    f"The city '{city}' conflicts with state '{state}'.",
                    f"Reference relation: {city} → {expected_state}.",
                    f"Replace state with '{expected_state.title()}'.",True,.99,'TXT-SEM-GEO-001'))
            if country and expected_country and country.lower()!=expected_country:
                issues.append(_issue(idx,'Text geographic contradiction','Error',lines[idx],
                    f"The city '{city}' conflicts with country '{country}'.",
                    f"Reference relation: {city} → {expected_country}.",
                    f"Replace country with '{expected_country.title()}'.",True,.99,'TXT-SEM-GEO-002'))

    # Entity-level contradictions: "Employee E101 ..." repeated with conflicting department/status/location.
    entity_pat=re.compile(r'\b(?:employee|customer|ticket|order|device|record|case)\s*(?:id\s*)?[:#-]?\s*([A-Z0-9-]{2,20})\b',re.I)
    field_pat=re.compile(r'\b(department|status|city|country|priority|resolution)\s*[:=-]\s*([^,;|.]+)',re.I)
    facts=defaultdict(lambda: defaultdict(list))
    for i,line in enumerate(lines):
        em=entity_pat.search(line)
        if not em: continue
        eid=em.group(1).upper()
        for fm in field_pat.finditer(line):
            field=fm.group(1).lower(); val=fm.group(2).strip()
            facts[eid][field].append((i,val))
    for eid,fields in facts.items():
        for field,vals in fields.items():
            canon=[STATUS_CANON.get(v.lower(),v.strip()) if field=='status' else v.strip() for _,v in vals]
            counts=Counter(v.lower() for v in canon)
            if len(counts)>1:
                winner,count=counts.most_common(1)[0]
                if count>=2:  # majority evidence required for auto semantic correction
                    exemplar=next(v for v in canon if v.lower()==winner)
                    for (row,raw),cv in zip(vals,canon):
                        if cv.lower()!=winner:
                            issues.append(_issue(row,'Cross-sentence semantic contradiction','Error',lines[row],
                                f"Entity {eid} has conflicting {field} values across the document.",
                                f"The value '{exemplar}' appears {count} times for {eid}; '{raw}' is the minority contradiction.",
                                f"Replace {field} '{raw}' with '{exemplar}'.",True,.97,'TXT-SEM-ENTITY-001'))

    return issues


def repair_text_dataset(df: pd.DataFrame):
    original=df.copy(); out=df.copy(); log=[]
    before=detect_text_issues(out)
    lines=out['text'].fillna('').astype(str).tolist()

    # Prepare entity-majority facts for context-aware repairs.
    entity_pat=re.compile(r'\b(?:employee|customer|ticket|order|device|record|case)\s*(?:id\s*)?[:#-]?\s*([A-Z0-9-]{2,20})\b',re.I)
    field_pat=re.compile(r'\b(department|status|city|country|priority|resolution)\s*[:=-]\s*([^,;|.]+)',re.I)
    facts=defaultdict(lambda: defaultdict(list))
    for i,line in enumerate(lines):
        em=entity_pat.search(line)
        if em:
            eid=em.group(1).upper()
            for fm in field_pat.finditer(line): facts[eid][fm.group(1).lower()].append((i,fm.group(2).strip()))

    drop=set()
    seen={}
    for i,line in enumerate(lines):
        old=line; new=line
        n=re.sub(r'\s+',' ',line.strip().lower())
        if n and n in seen:
            drop.add(i)
            log.append({'row':i,'column':'text','original_value':old,'repaired_value':'[duplicate removed]','issue_type':'Duplicate text line','reason':f'Exact normalized duplicate of line {seen[n]+1}.','confidence':.99,'confidence_basis':'Exact normalized equality.','status':'Automatically repaired','rule_id':'TXT-UNI-001'})
            continue
        if n: seen[n]=i
        new=re.sub(r'\s+',' ',new.strip())
        for bad,good in SPELLING.items():
            new=re.sub(rf'\b{re.escape(bad)}\b',good,new,flags=re.I)
        # email normalization
        new=re.sub(r'\s*(?:\[at\]|\(at\)| at )\s*','@',new,flags=re.I)
        new=re.sub(r'\s*(?:\[dot\]|\(dot\)| dot )\s*','.',new,flags=re.I)

        # geographic repair when city and state/country occur on same line
        cm=re.search(r'\bcity\s*[:=-]\s*([A-Za-z ]{2,40}?)(?=\s*(?:,|;|\||\bstate\b|\bcountry\b|$))',new,re.I)
        if cm:
            city=cm.group(1).strip(); c=city.lower()
            if c in STATE_CITY:
                sm=re.search(r'(\bstate\s*[:=-]\s*)([A-Za-z ]{2,40}?)(?=\s*(?:,|;|\||\bcountry\b|$))',new,re.I)
                if sm and sm.group(2).strip().lower()!=STATE_CITY[c]:
                    new=new[:sm.start(2)]+STATE_CITY[c].title()+new[sm.end(2):]
            if c in COUNTRY_CITY:
                km=re.search(r'(\bcountry\s*[:=-]\s*)([A-Za-z ]{2,40}?)(?=\s*(?:,|;|\||$))',new,re.I)
                if km and km.group(2).strip().lower()!=COUNTRY_CITY[c]:
                    new=new[:km.start(2)]+COUNTRY_CITY[c].title()+new[km.end(2):]

        # majority entity semantic correction
        em=entity_pat.search(new)
        if em:
            eid=em.group(1).upper()
            for field,vals in facts[eid].items():
                canon=[STATUS_CANON.get(v.lower(),v.strip()) if field=='status' else v.strip() for _,v in vals]
                counts=Counter(v.lower() for v in canon)
                if len(counts)>1:
                    winner,count=counts.most_common(1)[0]
                    if count>=2:
                        exemplar=next(v for v in canon if v.lower()==winner)
                        pat=re.compile(rf'(\b{re.escape(field)}\s*[:=-]\s*)([^,;|.]+)',re.I)
                        fm=pat.search(new)
                        if fm:
                            current=fm.group(2).strip()
                            if current.lower()!=winner:
                                new=new[:fm.start(2)]+exemplar+new[fm.end(2):]

        if new!=old:
            log.append({'row':i,'column':'text','original_value':old,'repaired_value':new,'issue_type':'Semantic/text normalization','reason':'Applied deterministic spelling, formatting, semantic-reference, or cross-sentence majority evidence.','confidence':.97,'confidence_basis':'Only deterministic or repeated-document evidence is used for automatic changes.','status':'Automatically repaired','rule_id':'TXT-HEAL-001'})
        lines[i]=new

    out['text']=lines
    if drop:
        out=out.drop(index=sorted(drop)).reset_index(drop=True)
        out['line_no']=range(1,len(out)+1)
    post=detect_text_issues(out)
    bad_rows=sorted(set(i.row for i in post if i.severity=='Error'))
    review=[]
    for idx in bad_rows:
        d=out.loc[idx].to_dict(); d['_review_reason']='; '.join(i.issue_type+': '+i.reason for i in post if i.row==idx and i.severity=='Error')
        d['_review_status']='Review required — semantic evidence insufficient'; d['_review_confidence']='; '.join(str(i.confidence) for i in post if i.row==idx and i.severity=='Error')
        review.append(d)
    review_df=pd.DataFrame(review)
    # Keep text lines in the cleaned document; semantic uncertainty should be visible,
    # not silently erase prose from the exported file.
    cleaned=out.reset_index(drop=True)
    remaining=detect_text_issues(cleaned)
    return cleaned,before,remaining,log,review_df,post
