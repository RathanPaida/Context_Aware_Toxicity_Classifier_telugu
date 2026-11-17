import re

def main():
    print("=== VALIDATING LATEX CODE, TABLES & CITATIONS ===")
    with open("paper_submission/main.tex", encoding="utf-8") as f:
        main_tex = f.read()
    with open("paper_submission/results_tables.tex", encoding="utf-8") as f:
        tables_tex = f.read()
    with open("paper_submission/references.bib", encoding="utf-8") as f:
        bib_tex = f.read()
    
    # 1. Check citations
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_tex))
    print(f"BibTeX keys defined ({len(bib_keys)}):", sorted(bib_keys))
    cited_keys = set()
    for m in re.finditer(r"\\cite[a-z]*\{([^}]+)\}", main_tex):
        for k in m.group(1).split(","):
            cited_keys.add(k.strip())
    print(f"Keys cited in main.tex ({len(cited_keys)}):", sorted(cited_keys))
    missing_cites = cited_keys - bib_keys
    if missing_cites:
        print("ERROR: Missing citation keys:", missing_cites)
    else:
        print("All citations resolved: OK")

    # 2. Check table labels and references
    labels = set(re.findall(r"\\label\{([^}]+)\}", tables_tex))
    print(f"Table labels defined ({len(labels)}):", sorted(labels))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", main_tex))
    print(f"Labels referenced in main.tex ({len(refs)}):", sorted(refs))
    missing_refs = refs - labels
    unreferenced = labels - refs
    if missing_refs:
        print("ERROR: Missing label targets in main.tex:", missing_refs)
    else:
        print("All referenced labels exist: OK")
    if unreferenced:
        print("WARNING: Defined labels not referenced:", unreferenced)
    else:
        print("All defined table labels referenced: OK")

    # 3. Check tabular column alignment and cell counts with proper brace tracking
    pos = 0
    errors = 0
    table_idx = 1
    while True:
        start = tables_tex.find(r"\begin{tabular}", pos)
        if start == -1:
            break
        brace_start = tables_tex.find("{", start + len(r"\begin{tabular}"))
        depth = 0
        i = brace_start
        while i < len(tables_tex):
            if tables_tex[i] == "{":
                depth += 1
            elif tables_tex[i] == "}":
                depth -= 1
                if depth == 0:
                    brace_end = i
                    break
            i += 1
        spec = tables_tex[brace_start+1:brace_end]
        end_tabular = tables_tex.find(r"\end{tabular}", brace_end)
        body = tables_tex[brace_end+1:end_tabular]
        
        col_tokens = re.findall(r"[lcr]|p\{[^}]*\}", spec)
        expected_cols = len(col_tokens)
        print(f"Table {table_idx}: spec='{spec}' -> expected {expected_cols} cols")
        
        for line_num, line in enumerate(body.strip().split("\n"), 1):
            clean = line.strip()
            if not clean or any(clean.startswith(c) for c in [r"\toprule", r"\midrule", r"\bottomrule", r"\multicolumn"]):
                continue
            cells = [c.strip() for c in clean.replace(r"\&", "").rstrip(r"\\").split("&")]
            if len(cells) != expected_cols:
                print(f"  ERROR Table {table_idx} line {line_num}: {len(cells)} cells, expected {expected_cols}: {clean[:60]}")
                errors += 1
        pos = end_tabular + len(r"\end{tabular}")
        table_idx += 1

    if errors == 0:
        print("All tabular columns & cell counts: OK")
    else:
        print(f"Total tabular cell count errors: {errors}")

if __name__ == "__main__":
    main()
