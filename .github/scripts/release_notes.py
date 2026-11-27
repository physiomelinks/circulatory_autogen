"""Write a version's GitHub release notes from CHANGELOG.md.

Usage::

    python3 .github/scripts/release_notes.py VERSION CHANGELOG.md notes.md

The notes are the version's section (from its ``## VERSION`` heading to the next ``## ``
heading) and an install footer. GitHub refuses a release body over ``LIMIT`` characters, so a
longer section keeps its intro and as many of its ``### `` subsections, in order, as fit; the
rest are listed by heading with a link to the full section at the version's tag. An intro that
does not fit by itself is cut at a line end. A version with
no section gets a stub pointing at CHANGELOG.md.
"""
import re
import sys

#: GitHub's maximum release body, in characters.
LIMIT = 125000

REPO_URL = 'https://github.com/physiomelinks/circulatory_autogen'

#: Marks where an intro too long to fit was cut.
CUT = '\n(cut here to fit)\n\n'


def section(changelog, version):
    """The heading and body of ``version``'s section, or ``(None, '')`` when it has none."""
    heading, body = None, []
    for line in changelog.splitlines(keepends=True):
        if heading is None:
            if re.match(r'^## ' + re.escape(version) + r'( |$|—)', line):
                heading = line[3:].strip()
        elif line.startswith('## '):
            break
        else:
            body.append(line)
    return heading, ''.join(body)


def anchor(heading):
    """GitHub's anchor for a markdown heading."""
    return re.sub(r'[^\w\- ]', '', heading.lower()).replace(' ', '-')


def footer(version):
    return (f'\n---\n\nInstall: `pip install libcuflynx=={version}`\n'
            f'PyPI: https://pypi.org/project/libcuflynx/{version}/\n')


def shorten(body, link, limit):
    """``body`` cut to its intro and the leading ``### `` subsections that fit in ``limit``,
    then a list of the remaining subsections' headings and ``link``."""
    parts = re.split(r'(?m)^(?=### )', body)
    kept, subsections = parts[0], parts[1:]
    headings = [s.splitlines()[0][4:] for s in subsections]
    rest = _rest_list(headings or ['The rest of the intro'], link)
    if len(kept) + len(rest) > limit:
        # An intro that does not fit by itself is cut at a line end, and every subsection listed.
        cut = kept.rfind('\n', 0, limit - len(rest) - len(CUT)) + 1
        return kept[:cut] + CUT + rest
    for i, sub in enumerate(subsections):
        if len(kept) + len(sub) + len(_rest_list(headings[i + 1:], link)) > limit:
            return kept.rstrip('\n') + '\n\n' + _rest_list(headings[i:], link)
        kept += sub
    return kept


def _rest_list(headings, link):
    if not headings:
        return ''
    items = ''.join(f'- {h}\n' for h in headings)
    return (f'### Everything else\n\nThe full notes are longer than a GitHub release allows. '
            f'They are in [CHANGELOG.md]({link}), under these headings:\n\n{items}')


def notes(changelog, version, limit=LIMIT):
    heading, body = section(changelog, version)
    end = footer(version)
    if heading is None:
        return 'See CHANGELOG.md.\n' + end
    if len(body) + len(end) <= limit:
        return body + end
    link = f'{REPO_URL}/blob/v{version}/CHANGELOG.md#{anchor(heading)}'
    return shorten(body, link, limit - len(end)) + end


def main(argv):
    version, changelog_path, out_path = argv
    with open(changelog_path, encoding='utf-8') as f:
        text = notes(f.read(), version)
    if text.startswith('See CHANGELOG.md.'):
        print(f'::warning::no CHANGELOG.md section for {version}; publishing a stub')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
