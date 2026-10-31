"""Build docs/index.html from docs/article.md (run after editing the article: python docs/build_page.py).

The page uses only tags Medium's story importer keeps (h1-h3, p, figure/img/figcaption, ul/li, blockquote,
pre/code, a, em, strong), so the published GitHub Pages URL can be imported into Medium directly.
"""
import os
import re

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
md = open(os.path.join(HERE, "article.md"), encoding="utf-8").read()
body = markdown.markdown(md, extensions=["fenced_code"])
# standalone images -> <figure> with caption (Medium keeps figure/figcaption)
body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)" ?/?></p>',
              lambda m: f'<figure><img src="{m.group(2)}" alt="{m.group(1)}"><figcaption>{m.group(1)}</figcaption></figure>',
              body)
title = re.search(r"<h1>(.*?)</h1>", body).group(1)
desc = re.search(r"<h2>(.*?)</h2>", body).group(1)
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<meta name="author" content="Angshul Majumdar">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta property="og:type" content="article">
<meta property="og:image" content="images/01-pipeline.png">
<link rel="stylesheet" href="styles.css">
</head>
<body>
<article>
{body}
</article>
</body>
</html>
"""
open(os.path.join(HERE, "index.html"), "w", encoding="utf-8").write(html)
print("wrote docs/index.html")
