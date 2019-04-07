import speech_recognition as sr
import time
import os

book = 'treasure'  # ['treasure', 'emma', 'huck', 'sherlock']
start = 1  # First utterance to process (leave at one unless code bugged out)
limit = float('inf')  # Maximum number of utterances to process (leave at infinity if not testing code)
record = True  # Set to False if results should not be recorded, otherwise True
sleep_list = [1, 5, 10, 30, 60, 300, 600]  # List of number of seconds to sleep before trying Google API again (if connection error)

# Load utterance attributes
rootdir = './' + book
utt_attributes = {}
with open(rootdir + '/txt.clean', 'r') as f:
    for line in f:
        (key, val) = line.strip().split('\t')
        utt_attributes[key] = val
print('Utterance attributes loaded. Beginning transcription.')

# Count number of files
total = 0
for _, _, files in os.walk(rootdir + '/wav'):
    for _ in files:
        total += 1

# Process and save STT
t0 = time.time()
r = sr.Recognizer()
i = 0
j = 0

for parent_dir, child_dir, files in os.walk(rootdir + '/wav'):
    for file in files:

        # Skip utterances as desired
        i += 1
        if i < start:
            continue

        # Exit process if reached limit
        if i >= start + limit:
            break

        # Process file
        full_path = os.path.join(parent_dir, file)
        file = sr.AudioFile(full_path)
        with file as source:
            file = r.record(source)

        # Send to Google API
        sleep = 0
        while sleep < len(sleep_list):
            try:
                results = r.recognize_google(file, show_all=True)
                break
            except:
                print('Google API cannot be reached. Trying again after',
                      sleep_list[sleep], 'second(s).')
                time.sleep(sleep_list[sleep])
                sleep += 1
        else:
            print('Error! Terminating process.')
            i = start + limit  # Will terminate in exit process condition above
            break

        # Adjust if empty list from Google API
        try:
            results = [i for i in results['alternative']]
        except:
            results = [{'transcript': ''}]

        # Process results from Google API
        transcripts = [i['transcript'].lower() for i in results]

        confidences = []
        for result in results:
            try:
                confidences.append(result['confidence'])
            except:
                confidences.append(0)

        split_path = full_path.split(os.path.sep)
        speaker = split_path[-2]
        utt = split_path[-1]
        ground_truth = utt_attributes[utt.split('.')[0]].lower()

        # Write to file
        if record:
            with open(book + '.txt', 'a') as f:
                f.write('\n' + str(i) + '\t'
                        + book + '\t'
                        + speaker + '\t'
                        + utt + '\t'
                        + '|'.join(transcripts) + '\t'
                        + '|'.join(str(i) for i in confidences)
                        + '\t' + ground_truth)
                f.close()

        # Update progress tracker
        t1 = time.time()
        print(i - start + 1, 'of', min(limit, total - start + 1),
              'utterances processed.',
              int(t1 - t0), 'seconds elapsed.',
              int(((t1 - t0) / (i - start + 1) * (
              min(limit, total - start + 1) - (i - start + 1))) / 60),
              'minutes left. Current utterance:',
              i, book, speaker, utt)

    else:
        continue
