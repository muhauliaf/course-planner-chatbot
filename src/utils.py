import re

import pypandoc
import requests
from bs4 import BeautifulSoup


def url_to_html(url):
    response = requests.get(url, verify=False)
    return response.text

def soup_strip_attributes(soup, use_whitelist=True):
    whitelist = ['a','img']
    for tag in soup.find_all(True):
        if not use_whitelist:
            tag.attrs = {}
        elif tag.name not in whitelist:
            tag.attrs = {}
        else:
            attrs = dict(tag.attrs)
            for attr in attrs:
                if attr not in ['src','href']:
                    del tag.attrs[attr]
    return soup

def url_to_md(url, noattr=True, minify=True):
    print('URL:',url)
    html_text = url_to_html(url)
    soup = BeautifulSoup(html_text, 'html.parser')
    if noattr:
        soup = soup_strip_attributes(soup)
    md_text = pypandoc.convert_text(str(soup), 'md', format='html')
    if minify:
        md_text = md_text.replace('<div>','').replace('</div>','')
        md_text = re.sub(r'[\r\n]+', '\n', md_text)
        md_text = re.sub(r" \s+", "  ",md_text)
        md_text = md_text.strip()
    print(f'Compression: {len(md_text)}/{len(html_text)} ({len(md_text)*100.0/len(html_text)})')
    md_text = md_text + f'\n\nSource Link URL : {url}'
    return md_text