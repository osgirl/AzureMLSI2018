from bs4 import BeautifulSoup
import os

path = '/Users/ephraimsalhanick/Desktop/AzureMLSI2018/Trump_Speeches'

for filename in os.listdir(path):
	
	with open(path + '/' + filename, "rb") as f:
    		soup = BeautifulSoup(f)
	
	# kill all script and style elements
	for script in soup(["script", "style"]):
    		script.extract()    # rip it out

	# get text
	text = soup.get_text()
		
	with open(path + '/' + filename.replace('htm','txt'), "w") as file_out:
		file_out.write(text)
			
