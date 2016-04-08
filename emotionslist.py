# coding: utf-8

from collections import OrderedDict

'''
The root emotions are manually collected for each emotion category
extra emotions could also be added if it suits the category
'''

happy_list = u"""happy glück smile frieden lol awesome lustig
lächerlich party unterhaltung""".split()

sad_list = u"""sad cry tragisch trauma unzufrieden trauer
traurig enttäuscht einsam tieftraurig flüchtlinge peinlich""".split()

angry_list = u"""stress zorniger annoyed nervig pissed angry anger
irritation depression wütend böse beleidigt lästig unerträglich ärgerlich""".split()

disgust_list = u"""ekel eklig scheiße unangenehm hass hässlich belastend
widerwärtig widerlich abscheulich pfui shit crap""".split()

surprise_list = u"""surprise wtf amazing omg wow unbelievable
überraschung wunder erstaunlich unerwartet unglaublich""".split()

fear_list = u"""fear panic panik furcht grauen risiko danger risk anxiety death angst
beängstigend hazard parisattacks bombe terror""".split()

root_emotions_categorised = [happy_list, sad_list, angry_list, disgust_list, surprise_list, fear_list]
root_emotions_container = [emotions for category in root_emotions_categorised for emotions in category]

emotion_categories = OrderedDict()
emotion_categories = OrderedDict([
    ('Happy' , 'cyan'),
    ('Sad' , 'blue'),
    ('Angry' , 'violet'),
    ('Disgust' , 'green'),
    ('Surprise' , 'orange'),
    ('Fear' , 'brown')])
