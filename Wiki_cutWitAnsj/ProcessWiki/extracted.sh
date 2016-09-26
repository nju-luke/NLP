python WikiExtractor.py -b 2000M -o extracted >output.txt zhwiki-latest-pages-articles.xml.bz2
opencc -i extracted/AA/wiki_00 -o extracted/AA/wiki00_chs -c zht2zhs.ini
opencc -i extracted/AA/wiki_01 -o extracted/AA/wiki01_chs -c zht2zhs.ini
opencc -i extracted/AA/wiki_02 -o extracted/AA/wiki02_chs -c zht2zhs.ini


