import codecs, os

class MySentences(object):
    
    def __init__(self, file_name):
        self.file_name = file_name
    
    def __iter__(self):
        for lines in codecs.open(self.file_name, 'r', 'utf-8'):
            yield lines.split()

class HashtagsExtraction(object):

	def __init__(self, file_name):
		self.file_name = file_name

	def extractHashtags(self):
		base,ext = os.path.splitext(self.file_name)
		write_file = open(self.file_name.rstrip(ext) + '_h' + ext ,'w')
		self.sentences = MySentences(self.file_name)
		for line in self.sentences:
			for word in line:
				if word.startswith('#'):
					write_file.write(word)
					write_file.write('\n')
		write_file.close()