# coding: utf-8

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from extracthashtags import HashtagsExtraction
import codecs

source_file = '/path/to/file/containing/tweets'

'''
hashtags are only extracted from the file containing tweets
'''
extracted_hashtags = HashtagsExtraction(source_file).extractHashtags()
hashtags_text = codecs.open((extracted_hashtags),'r','utf-8').read().splitlines()


def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(0, 60)


formatted_text = []
samples = 2000
for ic,l in enumerate(hashtags_text[:samples]):
    try:
        formatted_text.append(l.decode('unicode-escape'))
    except:
        pass


formatted_text = ' '.join(formatted_text)

'''
WordCloud library is used to plot the word cloud
'''
wc = WordCloud(
                      font_path = '/home/sivasurya/fonts/steelfish.ttf',
                      background_color='white',
                      min_font_size=25,
                      max_font_size=250,
                      random_state= 10,
                      width=1200,
                      height=800
                     ).generate(formatted_text)


default_colors = wc.to_array()
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3))
wc.to_file("hashtags_cloud.png")

plt.axis("off")
plt.show()