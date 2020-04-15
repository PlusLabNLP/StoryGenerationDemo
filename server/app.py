#! /usr/bin/env python3

import torch.multiprocessing as mp

import argparse

from flask import Flask
from flask import jsonify
from flask import session
from flask import request
from flask import make_response
from flask import render_template
from flask_pymongo import PyMongo
from flask_session import Session
from flask_cors import CORS

import uuid
import dateutil.parser
from datetime import datetime

import os
import io
import csv
import time

#import system1_beam as system1
#import system2
#import system3

app=Flask(__name__)
CORS(app) # Make the browser happier about cross-domain references.
app.secret_key = 'a20bec8e-9eb3-4f8f-a3cd-c80fb0fb9f3f'

# Add mongo db settings for logging
MONGO_URI = os.environ.get('MONGO_URI') or 'mongodb://0.0.0.0:27017/cwc'
app.config["MONGO_URI"] = MONGO_URI
mongo = PyMongo(app)

# Initialize the flask session
SESSION_TYPE = 'mongodb'
SESSION_MONGODB = mongo.cx
SESSION_MONGODB_DB = 'cwc'
SESSION_MONGODB_COLLECT = 'sessions'
SESSION_USE_SIGNER = True
app.config.from_object(__name__)
Session(app)

# Tell the browser to discard cached static files after
# 300 seconds.  This will facilitate rapid development
# (for even more rapid development, set the value lower).
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300

AUTOMATIC_HTML_FILE="index.html"
INTERACTIVE_HTML_FILE="interactive.html"

SYSTEM_1_ID = "system_1"
SYSTEM_2_ID = "system_2"

# Future advanced interface options:
KW_TEMP = 0.5
STORY_TEMP = None

# The server debug mode state.
SERVER_DEBUG_MODE = False


@app.route("/")
def read_root():
    """Supply the top-level HTML file."""
    ts = datetime.now().isoformat()
    uid = uuid.uuid4()
    session['ts'] = ts
    session['uid'] = uid
    return render_template(AUTOMATIC_HTML_FILE, SERVER_DEBUG_MODE=SERVER_DEBUG_MODE)


@app.route("/index.html")
def read_index():
    """Supply the top-level HTML file."""
    ts = datetime.now().isoformat()
    uid = uuid.uuid4()
    session['ts'] = ts
    session['uid'] = uid
    return render_template(AUTOMATIC_HTML_FILE, SERVER_DEBUG_MODE=SERVER_DEBUG_MODE)


@app.route("/interactive.html")
def read_interactive():
    """Supply the interactive HTML file."""
    ts = datetime.now().isoformat()
    uid = uuid.uuid4()
    session['ts'] = ts
    session['uid'] = uid
    return render_template(INTERACTIVE_HTML_FILE, SERVER_DEBUG_MODE=SERVER_DEBUG_MODE)


@app.route("/api/actions", methods=["POST"])
def action():
    data = request.json
    uid = session.get('uid')
    ts = datetime.now().isoformat()

    mongo.db.actions.insert_one({'ts': ts, 'session': uid, **data})

    # remove session id if user decides to 'reset'
    if data.get('category') == 'reset':
        uid = uuid.uuid4()
        session['ts'] = ts
        session['uid'] = uid

    return jsonify({'status': 'ok'})


@app.route("/api/logs", methods=["GET"])
def logs():
    uid = session.get('uid')
    logs = mongo.db.actions.find({'session': uid}).sort('ts')

    download = request.args.get('download')
    if download:
        now = datetime.now()
        dest = io.StringIO()
        writer = csv.writer(dest)
        if download == 'formatted':
            filename = 'session_logs_formatted_{}.csv'.format(now.strftime("%H_%M_%S_%d_%b_%Y"))
            writer.writerow([
                'timestamp',
                'session_id',
                'user_id',
                'system_id',
                'storyline_temp (list of floats)',
                'story_temp (list of floats)',
                'title',
                'storyline',
                'initial_text (str)',
                'final_text (str)',
                'duration',
                'initial_wc (int)',
                'final_wc (int)',
                'type',
                'edited_segment (list of ints)',
                'editor',
            ])

            title = ''
            storyline = []
            system = 'system_3'
            kw_temp = KW_TEMP
            story_temp = STORY_TEMP

            new_story = []
            prev_story = []
            prev_timestamp = None

            for log in logs:
                state_changed = False
                edited_segment = ''
                editor = log.get('editor', '')
                category = log.get('category', '')
                subcategory = log.get('subcategory', '')
                etype = log.get('type', '')

                # set the new formatted story to a copy of the previous one
                new_story = prev_story.copy()

                if log.get('category') == 'start' and log.get('subcategory') == 'session':
                    title = log.get('params')

                # Format changes to the settings
                if category == 'settings':
                    if subcategory == 'kw_temp':
                        kw_temp = log.get('params', kw_temp)
                    if subcategory == 'story_temp':
                        story_temp = log.get('params', story_temp)
                    if subcategory == 'system_2' and log.get('params', False):
                        system = 'system_2'
                    if subcategory == 'system_3' and log.get('params', False):
                        system = 'system_3'

                # Format changes to the storylines (phrases)
                if editor == 'system' and etype == 'generate':
                    state_changed = True
                    if category == 'phrase':
                        phrase = log.get('params')
                        storyline.append(phrase.split('. ')[1])
                    elif category == 'storyline':
                        storyline = log.get('params').split('->')
                if editor == 'human' and category == 'edit' and subcategory == 'phrase':
                    state_changed = True
                    phrase = log.get('params', '')
                    try:
                        edited_segment = int(phrase[0])
                        storyline[edited_segment-1] = phrase.split(' -> ')[1]
                    except IndexError:
                        storyline.append(phrase.split(' -> ')[1])

                # Format changes to the story (sentences)
                if editor == 'system' and etype == 'generate':
                    state_changed = True
                    if category == 'sentence':
                        sentence = log.get('params', '')
                        try:
                            edited_segment = int(sentence[0])
                            new_story[edited_segment-1] = sentence.split('. ')[1]
                        except IndexError:
                            new_story.append(sentence.split('. ')[1])
                    elif category == 'story':
                        new_story = log.get('params', '').split('<br>')

                if editor == 'human' and category == 'edit' and subcategory == 'sentence':
                    state_changed = True
                    sentence = log.get('params', '')
                    try:
                        edited_segment = int(sentence[0])
                        new_story[edited_segment-1] = sentence.split(' -> ')[1]
                    except IndexError:
                        new_story.append(sentence.split(' -> ')[1])

                ts = log.get('ts')
                timestamp = dateutil.parser.parse(ts)
                if state_changed:
                    # format the event type for the logs
                    event_type = '{}; {}; {}'.format(
                        etype,
                        category,
                        subcategory,
                    )

                    if prev_timestamp:
                        duration = timestamp - prev_timestamp
                    elif session['ts']:
                        duration = timestamp - dateutil.parser.parse(session['ts'])
                    else:
                        duration = 0

                    writer.writerow([
                        ts,
                        log.get('session'),
                        log.get('user'),
                        system,
                        kw_temp,
                        story_temp,
                        title,
                        ' -> '.join(storyline),
                        ' '.join(prev_story),
                        ' '.join(new_story),
                        duration,
                        sum([len(list(filter(None, s.split(' ')))) for s in prev_story]),
                        sum([len(list(filter(None, s.split(' ')))) for s in new_story]),
                        event_type,
                        edited_segment,
                        editor,
                    ])

                    prev_story = new_story
                prev_timestamp = timestamp
        else:
            filename = 'session_logs_raw_{}.csv'.format(now.strftime("%H_%M_%S_%d_%b_%Y"))
            writer.writerow(['#', 'session', 'user', 'mode', 'editor', 'type', 'category', 'subcategory', 'parameters', 'timestamp'])
            for i, log in enumerate(logs):
                writer.writerow([
                    i+1,
                    log.get('session'),
                    log.get('user'),
                    log.get('mode'),
                    log.get('editor'),
                    log.get('type'),
                    log.get('category'),
                    log.get('subcategory'),
                    log.get('params'),
                    log.get('ts'),
                ])
        response = make_response(dest.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename={}".format(filename)
        response.headers["Content-type"] = "text/csv"
        return response

    response = [
        {
            'mode': log.get('mode'),
            'editor': log.get('editor'),
            'type': log.get('type'),
            'category': log.get('category'),
            'subcategory': log.get('subcategory'),
            'parameters': log.get('params'),
            'timestamp': log.get('ts'),
        } for log in logs
    ]
    return jsonify(response)


@app.route("/api/generate", methods=["GET", "POST"])
def generate():
    request_id = request.values.get("id", "")
    topic = request.values.get("topic", "")
    systems_default = " ".join([ SYSTEM_1_ID, SYSTEM_2_ID, SYSTEM_3_ID ])
    systems = request.values.get("systems", systems_default).split()

    kw_temp = request.values.get("kw_temp", KW_TEMP)
    if str(kw_temp).upper() == "NONE":
        kw_temp = None
    elif str(kw_temp) == "":
        kw_temp = KW_TEMP
    else:
        kw_temp = float(kw_temp)
        if kw_temp == 0.0: # Protect against divide-by-zero.
            kw_temp = 0.001

    story_temp = request.values.get("story_temp", STORY_TEMP)
    if str(story_temp).upper() == "NONE":
        story_temp = None
    elif str(story_temp) == "":
        story_temp = STORY_TEMP
    else:
        story_temp = float(story_temp)
        if story_temp == 0.0: # Protect against divide-by-zero.
            story_temp = 0.001

    dedup = request.values.get("dedup", "")
    if dedup == "":
        dedup = True
    elif dedup.upper() == "TRUE":
        dedup = True
    else:
        dedup = False

    max_len = request.values.get("max_len", "")
    if max_len.upper() == "NONE":
        max_len = None
    elif max_len == "":
        max_len = None
    else:
        max_len = int(max_len)

    use_gold_titles = request.values.get("use_gold_titles", "NONE")
    if use_gold_titles.upper() == "NONE":
        use_gold_titles = None
    elif use_gold_titles == "":
        use_gold_titles = None
    elif use_gold_titles.upper() == "TRUE":
        use_gold_titles = True
    elif use_gold_titles.upper() == "FALSE":
        use_gold_titles = False

    # We'd like to record the elapsed clock time that
    # it takes to process this request.
    start_time = time.perf_counter() # replaces time.clock()

    # TODO: replace this to generate story for each one of the systems
    for system_id in systems:
        start_generation(system_id, topic, kw_temp, story_temp, dedup, max_len, use_gold_titles)

    response = {
        "n_story": request_id, # Nominally the number of stories generated.
        "storey_id": request_id # Nominally an ID for this story.
    }

    for system_id in systems:
        response[system_id] = get_response(system_id)

    # Record the end time, compute the elapsed seconds as a floating point
    # number, and format with two decimal points.
    end_time = time.perf_counter()
    response["elapsed"] = "{:.2f}".format(end_time - start_time)

    # create a separate system generated log item
    uid = session.get('uid')
    ts = datetime.now().isoformat()
    data = {
        'mode': 'automatic',
        'editor': 'system',
        'type': 'generate',
        'category': 'story',
        'subcategory': system_id,
        'params': response.get(system_id, ''),
    }
    mongo.db.actions.insert_one({'ts': ts, 'session': uid, **data})

    return jsonify(response)


@app.route("/api/generate_storyline", methods=["GET", "POST"])
def generate_storyline():
    request_id = request.values.get("id", "")
    topic = request.values.get("topic", "")
    systems_default = " ".join([ SYSTEM_1_ID, SYSTEM_2_ID, SYSTEM_3_ID ])
    systems = request.values.get("systems", systems_default).split()

    kw_temp = request.values.get("kw_temp", KW_TEMP)
    if str(kw_temp).upper() == "NONE":
        kw_temp = None
    elif str(kw_temp) == "":
        kw_temp = KW_TEMP
    else:
        kw_temp = float(kw_temp)
        if kw_temp == 0.0: # Protect against divide-by-zero.
            kw_temp = 0.001

    dedup = request.values.get("dedup", "")
    if dedup == "":
        dedup = True
    elif dedup.upper() == "TRUE":
        dedup = True
    else:
        dedup = False

    max_len = request.values.get("max_len", "")
    if max_len.upper() == "NONE":
        max_len = None
    elif max_len == "":
        max_len = None
    else:
        max_len = int(max_len)

    use_gold_titles = request.values.get("use_gold_titles", "NONE")
    if use_gold_titles.upper() == "NONE":
        use_gold_titles = None
    elif use_gold_titles == "":
        use_gold_titles = None
    elif use_gold_titles.upper() == "TRUE":
        use_gold_titles = True
    elif use_gold_titles.upper() == "FALSE":
        use_gold_titles = False

    # print("Id=%s Topic=%s" % (request_id, topic))

    # We'd like to record the elapsed wall clock time that
    # it takes to process this request.
    start_time = time.perf_counter() # replaces time.clock()

    # TODO: replace this to generate storyline for each one of the systems
    for system_id in systems:
        start_storyline_generation(system_id, topic, kw_temp, dedup, max_len, use_gold_titles)

    response = {
        "n_story": request_id, # Nominally the number of stories generated.
        "storey_id": request_id # Nominally an ID for this story.
    }
    for system_id in systems:
        response[system_id] = get_response(system_id)

    # Record the end time, compute the elapsed seconds as a floating point
    # number, and format with two decimal points.
    end_time = time.perf_counter()
    response["elapsed"] = "{:.2f}".format(end_time - start_time)

    # create a separate system generated log item
    uid = session.get('uid')
    ts = datetime.now().isoformat()
    for system_id in systems:
        data = {
            'mode': 'automatic',
            'editor': 'system',
            'type': 'generate',
            'category': 'storyline',
            'subcategory': system_id,
            'params': response.get(system_id, ''),
        }
        mongo.db.actions.insert_one({'ts': ts, 'session': uid, **data})

    return jsonify(response)


@app.route("/api/collab_storyline", methods=["GET", "POST"])
def collab_storyline():
    request_id = request.values.get("id", "")
    system_id = request.values.get("system_id", SYSTEM_2_ID)
    topic = request.values.get("topic", "")
    storyline = request.values.get("storyline", "")
    if len(storyline) > 0:
        current_storyline = storyline.split("->")
    else:
        current_storyline = [ ]

    kw_temp = request.values.get("kw_temp", KW_TEMP)
    if str(kw_temp).upper() == "NONE":
        kw_temp = None
    elif str(kw_temp) == "":
        kw_temp = KW_TEMP
    else:
        kw_temp = float(kw_temp)
        if kw_temp == 0.0: # Protect against divide-by-zero.
            kw_temp = 0.001

    dedup = request.values.get("dedup", "")
    if dedup == "":
        dedup = True
    elif dedup.upper() == "TRUE":
        dedup = True
    else:
        dedup = False

    max_len = request.values.get("max_len", "")
    if max_len.upper() == "NONE":
        max_len = None
    elif max_len == "":
        max_len = None
    else:
        max_len = int(max_len)

    start_time = time.perf_counter() # replaces time.clock()
    start_collab_storyline(system_id, topic, current_storyline, kw_temp, dedup, max_len)

    response = get_response(system_id)
    response["request_id"] = request_id

    # Record the end time, compute the elapsed seconds as a floating point
    # number, and format with two decimal points.
    end_time = time.perf_counter()
    response["elapsed"] = "{:.2f}".format(end_time - start_time)

    # create a separate system generated log item
    uid = session.get('uid')
    phrase = '{}. {}'.format(
        len(current_storyline) + 1,
        response.get('new_phrase', ''),
    )
    ts = datetime.now().isoformat()
    data = {
        'mode': 'interactive',
        'editor': 'system',
        'type': 'generate',
        'category': 'phrase',
        'subcategory': system_id,
        'params': phrase,
    }
    mongo.db.actions.insert_one({'ts': ts, 'session': uid, **data})

    return jsonify(response)


@app.route("/api/generate_interactive_story", methods=["GET", "POST"])
def generate_interactive_story():
    request_json = request.get_json(force=True)
    request_id = request_json.get('id')
    system_id = request_json.get('system_id')
    topic = request_json.get('topic')
    storyline = request_json.get('storyline')
    use_sentence = request_json.get('use_sentence')
    sentences = request_json.get('sentences')
    change_max_idx = request_json.get('change_max_idx')
    only_one = request_json.get('only_one')

    # request_id = request.values.get("id", "")
    # system_id = request.values.get("system_id", SYSTEM_2_ID)
    # topic = request.values.get("topic", "")
    # storyline = request.values.get("storyline", "")
    if len(storyline) > 0:
        storyline_phrases = storyline.split("->")
    else:
        storyline_phrases = []

    print(request_json)
    story_temp = request_json.get("story_temp", STORY_TEMP)
    if str(story_temp).upper() == "NONE":
        story_temp = None
    elif str(story_temp) == "":
        story_temp = STORY_TEMP
    else:
        story_temp = float(story_temp)
        if story_temp == 0.0: # Protect against divide-by-zero.
            story_temp = 0.001

    print(story_temp)

    start_time = time.perf_counter() # replaces time.clock()

    # todo: replace this to generate interactive story for the given system type
    start_generate_interactive_story(system_id, topic, storyline_phrases, story_temp, use_sentence, sentences, change_max_idx, only_one)

    response = get_response(system_id)
    response["request_id"] = request_id

    # Record the end time, compute the elapsed seconds as a floating point
    # number, and format with two decimal points.
    end_time = time.perf_counter()
    response["elapsed"] = "{:.2f}".format(end_time - start_time)
    print(response)

    # create a separate system generated log item
    uid = session.get('uid')
    story = response.get('story', '').split(' </s> ')
    sentence = '{}. {}'.format(len(story), story[-1])
    if only_one:
        data = {
            'mode': 'interactive',
            'editor': 'system',
            'type': 'generate',
            'category': 'sentence',
            'subcategory': system_id,
            'params': sentence,
        }
    else:
        data = {
            'mode': 'interactive',
            'editor': 'system',
            'type': 'generate',
            'category': 'story',
            'subcategory': system_id,
            'params': response.get('story', '').replace('</s>', '<br>'),
        }
    ts = datetime.now().isoformat()
    mongo.db.actions.insert_one({'ts': ts, 'session': uid, **data})

    return jsonify(response)


@app.route("/api/generate_story", methods=["GET", "POST"])
def generate_story():
    request_id = request.values.get("id", "")
    system_id = request.values.get("system_id", SYSTEM_2_ID)
    topic = request.values.get("topic", "")
    storyline = request.values.get("storyline", "")
    if len(storyline) > 0:
        storyline_phrases = storyline.split("->")
    else:
        storyline_phrases = [ ]

    story_temp = request.values.get("story_temp", STORY_TEMP)
    if str(story_temp).upper() == "NONE":
        story_temp = None
    elif str(story_temp) == "":
        story_temp = STORY_TEMP
    else:
        story_temp = float(story_temp)
        if story_temp == 0.0: # Protect against divide-by-zero.
            story_temp = 0.001

    # TODO: replace this to generate story for the given system type
    start_time = time.perf_counter() # replaces time.clock()
    start_generate_story(system_id, topic, storyline_phrases, story_temp)

    response = get_response(system_id)
    response["request_id"] = request_id

    # Record the end time, compute the elapsed seconds as a floating point
    # number, and format with two decimal points.
    end_time = time.perf_counter()
    response["elapsed"] = "{:.2f}".format(end_time - start_time)

    # create a separate system generated log item
    uid = session.get('uid')
    ts = datetime.now().isoformat()
    data = {
        'mode': 'automatic',
        'editor': 'system',
        'type': 'generate',
        'category': 'story',
        'subcategory': system_id,
        'params': response.get('story', ''),
    }
    mongo.db.actions.insert_one({'ts': ts, 'session': uid, **data})

    return jsonify(response)


# write auto mode data to txt
@app.route("/api/write_auto_txt", methods=["GET", "POST"])
def write_auto_txt():
    request_json = request.get_json(force=True)
    with open("auto_mode_logging.txt", "a") as out:
        out.write("New auto generation data:" + "\n")
        out.write("Story Topic:" + "\n")
        out.write(request_json.get('topic') + "\n")
        out.write("Storyline:" + "\n")
        out.write(request_json.get('storyline') + "\n")
        out.write("System1_story:" + "\n")
        out.write(request_json.get('system1_story') + "\n")
        out.write("System2_story:" + "\n")
        out.write(request_json.get('system2_story') + "\n")
        out.write("System3_story:" + "\n")
        out.write(request_json.get('system3_story') + "\n")

    return "success"


# write interactive mode data to txt
@app.route("/api/write_interactive_txt", methods=["GET", "POST"])
def write_interactive_txt():
    request_json = request.get_json(force=True)
    with open("interactive_mode_logging.txt", "a") as out:
        out.write("New interactive generation data:" + "\n")
        out.write("Story Topic:" + "\n")
        out.write(request_json.get('topic') + "\n")
        out.write("Storyline:" + "\n")
        out.write(request_json.get('storyline') + "\n")

        kw_temp = request_json.get('kw_temp')
        if kw_temp == "":
            kw_temp = "None"
        out.write("kw_temp: " + kw_temp + "\n")

        story_temp = request_json.get('story_temp')
        if story_temp == "":
            story_temp = "None"
        out.write("story_temp: " + story_temp + "\n")

        out.write("generate story:" + "\n")
        out.write(request_json.get('story') + "\n")

    return "success"



# Allow the three system generators to run in parallel.  Each system's
# response is an HTML string, which we send back to the parent using
# a Queue.
#
# This design creates a permanent process for each system, initializing
# the generator in that process.

request_queues = {}
result_queues = {}


def initialize_generator(story_generator_class, system_id):
    request_queue = mp.Queue()
    request_queues[system_id] = request_queue
    result_queue = mp.Queue()
    result_queues[system_id] = result_queue
    system_process = mp.Process(target=system_worker, args=(system_id, story_generator_class, request_queue, result_queue))
    system_process.start()


def initialize_generators():
    # Initializing the sentence generators
    pass


def start_generation(system_id, topic, kw_temp, story_temp, dedup, max_len, use_gold_titles):
    """Ask a system to start generating a storyline and story response."""
    worker_request = {
        "action": "generate",
        "topic": topic,
        "kw_temp": kw_temp,
        "story_temp": story_temp,
        "dedup": dedup,
        "max_len": max_len,
        "use_gold_titles": use_gold_titles
    }
    request_queues[system_id].put(worker_request)


def start_storyline_generation(system_id, topic, kw_temp, dedup, max_len, use_gold_titles):
    """Ask a system to start generating a storyline."""
    worker_request = {
        "action": "generate_storyline",
        "topic": topic,
        "kw_temp": kw_temp,
        "dedup": dedup,
        "max_len": max_len,
        "use_gold_titles": use_gold_titles
    }
    request_queues[system_id].put(worker_request)


def start_collab_storyline(system_id, topic, storyline, kw_temp, dedup, max_len):
    """Ask a system to collaboratively generate a storyline."""
    worker_request = {
        "action": "collab_storyline",
        "topic": topic,
        "storyline": storyline,
        "kw_temp": kw_temp,
        "dedup": dedup,
        "max_len": max_len
    }
    request_queues[system_id].put(worker_request)


def start_generate_interactive_story(system_id, topic, storyline, story_temp, use_sentence, sentences, change_max_idx, only_one):
    """Ask a system to start generating a story."""
    pass_sentences = []
    # append non-empty sentences to be used as a prefix for later story generation

    # so if the third sentence is deleted, change_max_idx will be 2, and i will be 0,1,2. Basically I think
    # it is the index beyond which we can generate new content.
    # so pass sentences will be variable length and contain everything to keep.
    # if a user modifies 0 and 3, 3 will be passed
    if use_sentence:
        for i in range(change_max_idx + 1):
            if len(sentences[i]) > 0:
                pass_sentences.append(sentences[i])

    worker_request = {
        "action": "generate_interactive_story",
        "topic": topic,
        "storyline": storyline,
        "story_sentences": pass_sentences,
        "story_temp": story_temp,
        "only_one": only_one
    }
    print(worker_request)
    request_queues[system_id].put(worker_request)


def start_generate_story(system_id, topic, storyline, story_temp):
    """Ask a system to start generating a story."""
    worker_request = {
        "action": "generate_story",
        "topic": topic,
        "storyline": storyline,
        "story_temp": story_temp
    }
    request_queues[system_id].put(worker_request)


def system_worker(system_id, story_generator_class, request_queue, result_queue):
        story_generator = story_generator_class(system_id)

        while True:
            worker_request = request_queue.get()
            action = worker_request["action"]
            if action == "generate":
                result = story_generator.generate_response(worker_request["topic"],
                                                           kw_temp=worker_request["kw_temp"],
                                                           story_temp=worker_request["story_temp"],
                                                           dedup=worker_request.get("dedup", None),
                                                           max_len=worker_request.get("max_len", None),
                                                           use_gold_titles=worker_request.get("use_gold_titles", None)
                )
            elif action == "generate_storyline":
                result = story_generator.generate_storyline(worker_request["topic"],
                                                            kw_temp=worker_request["kw_temp"],
                                                            dedup=worker_request.get("dedup", None),
                                                            max_len=worker_request.get("max_len", None),
                                                            use_gold_titles=worker_request.get("use_gold_titles", None)
                )
            elif action == "collab_storyline":
                result = story_generator.collab_storyline(worker_request["topic"], worker_request["storyline"],
                                                          kw_temp=worker_request.get("kw_temp", None),
                                                          dedup=worker_request.get("dedup", None),
                                                          max_len=worker_request.get("max_len", None))
            elif action == "generate_interactive_story":
                result = story_generator.generate_interactive_story(worker_request["topic"], worker_request["storyline"],
                                                        worker_request["story_sentences"], story_temp=worker_request.get("story_temp", None),
                                                                    only_one=worker_request["only_one"])
            elif action == "generate_story":
                result = story_generator.generate_story(worker_request["topic"], worker_request["storyline"],
                                                        story_temp=worker_request.get("story_temp", None))
            else:
                result = {
                    system_id: "internal error"
                }

            result_queue.put(result)


def get_response(system_id):
    """Returns a system's response as HTML."""
    response = result_queues[system_id].get()
    return response


if __name__ == '__main__':
    # By default, start an externally visible server (host='0.0.0.0')
    # on port 5000.  Set the environment variable CWC_SERVER_PORT to
    # change this.  For example, under bash or sh, you can start a
    # copy of the server on port 5001 with the following command:
    #
    # CWC_SERVER_PORT=5001 ./web_server.py
    #
    # You can also select the port by passing the "--port <port>"
    # option on the command line.  The command line --port option
    # overrides the CWC_SERVER_PORT environment variable.
    #
    # Similarly, the default host is "0.0.0.0", which means run the
    # server on all IP addresses available to the process.  This may
    # be changed with the CWC_SERVER_HOST environmen variable or the
    # "--host <host>" command line option.  One useful choice is
    # 127.0.0.1 (or "localhost"), which mmakes the server accessible
    # only to Web browsers running on the same system as the Web
    # server.
    #
    # 14-Sep-2018: On cwc-story.isi.edu, port number 80 has been
    # redirected to port 5000.  This allows an unprivileged user to
    # run the CWC Web server and make it visible on the default Web
    # port.
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    default_host = os.environ.get("CWC_SERVER_HOST", "0.0.0.0")
    default_port = int(os.environ.get("CWC_SERVER_PORT", "5006"))

    parser = argparse.ArgumentParser()
    parser.add_argument(      '--debug', help="Run in debug mode (less restrictive).", required=False, action='store_true')
    parser.add_argument(      '--host', help="The Web server host name or IP address", required=False, default=default_host)
    parser.add_argument('-p', '--port', help="The Web server port number (5000..5009)", required=False, type=int, default=default_port)
    args=parser.parse_args()

    # Force static files to timeout quickly to ease debugging.
    if args.debug:
        app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
        SERVER_DEBUG_MODE = True

    print("Initializing the sentence generators.")
    initialize_generators()

    print("Starting the server on host %s port %d" % (args.host, args.port))
    app.run(host=args.host, port=args.port, threaded=True, debug=SERVER_DEBUG_MODE)
