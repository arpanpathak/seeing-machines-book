#!/usr/bin/env python3
"""Convert all math delimiters in mdBook markdown files.
   $...$  ->  \( ... \)
   $$...$$ -> \[ ... \]
   This is needed because MathJax's default config only recognizes \(...\) and \[...\].
   In CommonMark markdown, backslash before punctuation is eaten, so we use \\
   to produce a literal backslash in HTML.
"""

import re, os

base = "/Users/arpanpathak/Projects/books/book/src"
files = []
for root, dirs, fnames in os.walk(base):
    for f in fnames:
        if f.endswith(".md"):
            files.append(os.path.join(root, f))

files.sort()
print(f"Found {len(files)} markdown files to process")


def is_latex_content(s: str) -> bool:
    """Check if text inside $...$ looks like LaTeX (has commands, subscripts, etc.)."""
    # Has LaTeX commands like \mathbf, \mathbb, \frac, etc.
    if re.search(r'\\[a-zA-Z]+', s):
        return True
    # Has subscripts/superscripts
    if '_' in s or '^' in s:
        return True
    # Has common math operators
    if re.search(r'[∑∫∂∇∈⊂⊆∪∩∀∃∞]', s):
        return True
    return False


total_inline = 0
total_display = 0

for filepath in files:
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # Step 1: Convert $$...$$ (display math) to \[...\]
    # Use a simple approach: match $$...$$ that spans lines
    content_new = ""
    i = 0
    while i < len(content):
        # Look for opening $$
        if content[i:i+2] == '$$':
            # Find closing $$
            j = content.find('$$', i+2)
            if j >= 0:
                inner = content[i+2:j]
                content_new += '\\\\[' + inner + '\\\\]'
                total_display += 1
                i = j + 2
                continue
        content_new += content[i]
        i += 1
    content = content_new

    # Step 2: Convert $...$ (inline math) to \(...\)
    content_new = ""
    i = 0
    while i < len(content):
        c = content[i]
        # Look for single $ (not preceded by $)
        if c == '$' and (i == 0 or content[i-1] != '$'):
            # Check next char isn't $ (avoid $$ which is display math)
            if i+1 < len(content) and content[i+1] == '$':
                content_new += '$'
                i += 1
                continue
            # Find closing $ (not followed by $)
            j = content.find('$', i+1)
            if j >= 0 and (j+1 >= len(content) or content[j+1] != '$'):
                inner = content[i+1:j]
                if is_latex_content(inner):
                    content_new += '\\\\( ' + inner + ' \\\\)'
                    total_inline += 1
                else:
                    content_new += '$' + inner + '$'
                i = j + 1
                continue
        content_new += c
        i += 1
    content = content_new

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  Modified: {os.path.relpath(filepath, base)}")

print(f"\nTotal conversions: {total_display} display equations, {total_inline} inline math expressions")
print("Done! All math now uses \\\\(...\\\\) and \\\\[...\\\\] delimiters")
