# -*- coding: utf-8 -*-

from __future__ import print_function

import requests
import lxml
from lxml import html
import time
import codecs
import os
import urlparse

def url_to_filename(url):
    scheme, _, host_path = url.partition("://")
    host, _, path = host_path.partition("/")
    filename = "".join(x for x in host + "-"+path if (x.isalnum() or x in "%.-_"))
#    print("%s -> %s" % (url, filename))
    return filename
def main():

    start_url = "https://sv.wikipedia.org/"
    lang = "sv"
    scrape_limit = 1000

    already_visited = set()
    to_visit = set()
    to_visit.add(start_url)

    wait_time = 0

    saved_page_count = 0
    while to_visit:
        url = to_visit.pop()

        time.sleep(wait_time)
        wait_time = 1
        print("Loading %s from pool of %d" % (url, len(to_visit) + 1))
        page = requests.get(url)
        redirected_to_url = page.url
        already_visited.add(url)
        if redirected_to_url:
            already_visited.add(redirected_to_url)

        page_tree = html.fromstring(page.content)
        page_tree.resolve_base_href()
        page_tree.make_links_absolute(url)

#        print(dir(page_tree))
#        print(page_tree.getchildren())
#        html_elm = page_tree.cssselect("html")
        page_lang = page_tree.attrib.get("lang")
#        print("lang: %s" % page_lang)
        if page_lang != lang:
            print("%s is the wrong language (%s) - skipping" % (url, page_lang))
            continue
        wikipedia_content_elm = page_tree.get_element_by_id("mw-content-text")
#        print(wikipedia_content_elm)
        content_as_text = wikipedia_content_elm.text_content()
        # Compress whitespace
        one_line_text = " ".join(content_as_text.split())
        print(one_line_text[:200].encode("utf-8"))

        for element, attribute, link, pos in wikipedia_content_elm.iterlinks():
            if attribute == "href":
                if "?" in link:
                    # Ignore things that look like queries.
                    continue
                if urlparse.urlparse(link).netloc != urlparse.urlparse(start_url).netloc:
                    # print("%s vs %s" % (link, start_url))
                    # print("%s vs %s" % (urlparse.urlparse(link).netloc,
                    #                     urlparse.urlparse(start_url).netloc))
                    # Don't escape this site.
                    continue
                if link not in already_visited:
                    to_visit.add(link)

        if redirected_to_url:
            filename = url_to_filename(redirected_to_url)
        else:
            filename = url_to_filename(url)
        with codecs.open(os.path.join("scrape_data", lang, filename), "w", "utf-8") as f:
            f.write(content_as_text)
            saved_page_count = saved_page_count + 1
            print("Saved (%d): %s" % (saved_page_count, filename))

        if saved_page_count >= scrape_limit:
            print("DONE - we got all we wanted")
            return

    print("DONE - ran out of links to follow")
#        print(to_visit)

if __name__ == "__main__":
    main()
