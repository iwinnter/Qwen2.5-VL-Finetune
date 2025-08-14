import json
import sys
import os
import urllib.request as ureq
import urllib.error # Import the error module
# import pdb # Commented out pdb as it's for debugging

download = 1  # 0 if images are already downloaded

###############################################################
######################### load dataset json file ###############
################################################################
try:
    with open('dataset.json', 'r') as fp:
        data = json.load(fp)
    print("✓ Successfully loaded dataset.json")
except FileNotFoundError:
    print("✗ Error: dataset.json not found in the current directory.")
    sys.exit(1)
except json.JSONDecodeError:
    print("✗ Error: dataset.json is not a valid JSON file.")
    sys.exit(1)


## dictionary data contains image URL, questions and answers ##

################################################################
############### Script for downloading images ##################
################################################################
## Make a directory images to store all images there ##########
if download == 1:
    os.makedirs('./images', exist_ok=True) # Use makedirs with exist_ok to avoid errors if folder exists
    print("Starting image download...")
    failed_downloads = 0
    total_urls = len(data.keys())
    
    for idx, k in enumerate(data.keys()):
        try:
            img_url = data[k]['imageURL']
            ext = os.path.splitext(img_url)[1]
            # Handle cases where URL might not have an extension
            if not ext:
                print(f"Warning: No extension found for URL {img_url}, defaulting to .jpg")
                ext = '.jpg'
            outputFile = f'images/{k}{ext}'
            
            print(f"({idx+1}/{total_urls}) Downloading: {img_url}") # Optional: Print progress
            ureq.urlretrieve(img_url, outputFile)
            # print(f"Downloaded: {outputFile}") # Optional: Print success
            
        except urllib.error.HTTPError as e:
            print(f"HTTP Error for key '{k}' (URL: {img_url}): {e.code} {e.reason}")
            failed_downloads += 1
        except urllib.error.URLError as e:
            print(f"URL Error for key '{k}' (URL: {img_url}): {e.reason}")
            failed_downloads += 1
        except Exception as e: # Catch other potential errors
            print(f"Unexpected error downloading image for key '{k}' (URL: {img_url}): {e}")
            failed_downloads += 1
            
    print(f"Image download completed. Failed downloads: {failed_downloads}")


#################################################################
################### Example of data access #####################
################################################################
print("\n--- Data Access Examples ---")
count = 0
max_examples = 3 # Limit examples printed
for k in data.keys():
    if count >= max_examples:
        break
    try:
        img_url = data[k]['imageURL']
        ext = os.path.splitext(img_url)[1]
        if not ext:
             ext = '.jpg' # Use default if missing
        imageFile = f'images/{k}{ext}'

        print('************************')
        print('Image file: %s' % (imageFile))
        print('List of questions:')
        print(data[k]['questions'])
        print('List of corresponding answers:')
        print(data[k]['answers'])
        print('Use this image as training (1), validation (2) or testing (3): %s' % (data[k]['split']))
        print('*************************')
        count += 1
    except Exception as e:
        print(f"Error accessing data for key '{k}': {e}")



######################################################################
########################### Get dataset stats ########################
######################################################################
print("\n--- Dataset Statistics ---")
genSet = set()
for k in data.keys():
    if 'genre' in data[k]: # Check if 'genre' key exists
        genSet.add(data[k]['genre'])
    else:
        print(f"Warning: 'genre' key missing for item {k}")


numImages = len(data.keys())
numQApairs = 0
numWordsInQuestions = 0
numWordsInAnswers = 0
numQuestionsPerImage = 0
ANS = set()  # Set of unique answers
authorSet = set()
bookSet = set()


for imgId in data.keys():
    # Check for existence of keys to avoid KeyError
    if 'questions' not in data[imgId] or 'answers' not in data[imgId]:
        print(f"Warning: Missing 'questions' or 'answers' for item {imgId}")
        continue

    numQApairs += len(data[imgId]['questions'])
    numQuestionsPerImage += len(data[imgId]['questions'])
    
    if 'authorName' in data[imgId]:
        authorSet.add(data[imgId]['authorName'])
    if 'title' in data[imgId]:
        bookSet.add(data[imgId]['title'])

    for qno in range(len(data[imgId]['questions'])):
        ques = data[imgId]['questions'][qno]
        if isinstance(ques, str): # Ensure question is a string
            numWordsInQuestions += len(ques.split())
        else:
             print(f"Warning: Question {qno} for item {imgId} is not a string: {ques}")

    for ano in range(len(data[imgId]['answers'])):
        ans = data[imgId]['answers'][ano]
        ANS.add(str(ans)) # Convert to string to ensure it's hashable
        numWordsInAnswers += len(str(ans).split())



print("--------------------------------")
print("Number of Images (entries in JSON): %d" % (numImages))
print("Number of QA pairs: %d" % (numQApairs))
print("Number of unique authors: %d" % (len(authorSet)))
print("Number of unique titles: %d" % (len(bookSet)))
print("Number of unique answers: %d" % (len(ANS)))
print("Number of unique genres: %d" % (len(genSet)))
if numQApairs > 0:
    print("Average question length (in words): %.2f" % (float(numWordsInQuestions) / float(numQApairs)))
    print("Average answer length (in words): %.2f" % (float(numWordsInAnswers) / float(numQApairs)))
else:
    print("Average question length (in words): N/A (No QA pairs)")
    print("Average answer length (in words): N/A (No QA pairs)")

if numImages > 0:
    print("Average number of questions per image: %.2f" % (float(numQuestionsPerImage) / float(numImages)))
else:
    print("Average number of questions per image: N/A (No images)")
print("--------------------------------")
