from lxml import html
import requests
import io
import time
import os
import random

for pg in range(1, 218):
    try:
        print(pg)
        page = requests.get('https://www.zaytung.com/digerleri.asp?pg=' + str(pg))
        pageTree = html.fromstring(page.content)

        table = pageTree.xpath('//div[@id="mainContent"]//h3/a/@href')
        print(table)
        time.sleep(random.randint(0, 5))

        for link in table:
            try:
                print(link)
                id = link[link.rfind('=') + 1:]

                if not os.path.isfile("./Data/Fake/" + id + ".txt"):
                    haber = requests.get('https://www.zaytung.com/' + link.strip())
                    haberTree = html.fromstring(haber.content)

                    title = haberTree.xpath('//div[@id="manset"]/div/h1/text()')
                    body = haberTree.xpath('//div[@id="manset"]/div/p/text()')
                    content = " ".join(title) + " " + " ".join(body)
                    if content.strip() != "":
                        with io.open("./Data/Fake/" + id + ".txt", 'w', encoding='utf8') as f:
                            f.write(content.strip())

                    time.sleep(random.randint(0, 5))
            except:
                print("hata oldu")
    except:
        print("hata oldu")
