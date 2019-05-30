import argparse
import json
import os
import pickle
import string as stringlib

import matplotlib
import matplotlib.pyplot as plt

from imadial import ImageEditRealUserInterface


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--session-pickle', type=str,
                        required=True, help="Path to session pickle")
    parser.add_argument('-c', '--config', type=str,
                        default='./config/deploy/rule.json', help="Configuration")
    parser.add_argument('-o', '--output-png', type=str,
                        required=True, help="Path to output figure")
    args = parser.parse_args()
    return args


def split_string_by_length(string, length):

    assert isinstance(string, str)
    splits = [string[i: i + length] for i in range(0, len(string), length)]

    PUNCS = stringlib.punctuation + " "

    for i, line in enumerate(splits):
        if i == 0:
            continue

        if splits[i-1][-1] not in PUNCS and splits[i][0] not in PUNCS:
            splits[i-1] += "-"
    return splits


def create_dialogue_figure(images_and_utterances, output_png):

    nimages = len(images_and_utterances)

    NCOLS = 5
    NROWS = int((nimages+NCOLS)/NCOLS)

    HEIGHT = 10 * NROWS
    WIDTH = 6 * NCOLS

    LINE_LENGTH = 29

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    total_turns = 0
    for i in range(nimages):
        index = i+1
        fig.add_subplot(NROWS, NCOLS, index)
        image, utts = images_and_utterances[i]

        caption_lines = []
        for utt in utts:
            caption_lines += split_string_by_length(utt, LINE_LENGTH)

        caption = '\n'.join(caption_lines)

        not_first = 1 if index > 1 else 0
        start_turn = total_turns + not_first
        total_turns += (len(utts)+not_first) // 2

        plt.title("Frame {} Turns {} to {}".format(
            index, start_turn, total_turns), fontsize=20)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)
        plt.xlabel(caption, horizontalalignment="left", x=0.0, fontsize=20)

    print(output_png)
    plt.savefig(output_png, bbox_inches='tight', pad_inches=0)


def load_from_pickle(filepath):
    with open(filepath, 'rb') as fin:
        obj = pickle.load(fin)
    return obj


def load_from_json(filepath):
    with open(filepath, 'r') as fin:
        obj = json.loads(fin.read())
    return obj


def main():
    args = parse_args()

    # Reproduce the dialogue session
    config_file = args.config
    config_json = load_from_json(config_file)

    # Dialogue System
    DialogueSystem = ImageEditRealUserInterface(config_file)
    DialogueSystem.reset()

    session_pickle = args.session_pickle
    session = load_from_pickle(session_pickle)

    # Dialogue Actions
    acts = session['acts']

    # Load Image
    image_id = session['image_id']
    image_dir = config_json['image_dir']
    image_path = os.path.join(image_dir, '{}.jpg'.format(image_id))

    DialogueSystem.open(image_path)

    # Reproduce the whole dialogue session
    images_and_utterances = []  # list of tuples

    image = DialogueSystem.get_image()
    utts = []

    for turn_id, act in enumerate(acts):
        if turn_id == 0:
            sys_utt = "Hi! This is an image editing chatbot.  How may I help you?"
            utts.append("System: " + sys_utt)
            continue

        usr_or_vis_act = act[0]

        usr_utt = usr_or_vis_act.get("user_utterance", "")

        if usr_utt:
            user_act = act[0]
            DialogueSystem.observe(user_act)
            sys_act = DialogueSystem.act()
            # Visualization
            #print("User:", usr_utt)
            utts.append("User: " + usr_utt)
        else:
            vis_act = act[0]
            vis_utt = vis_act.get("visionengine_utterance", "")
            DialogueSystem.observe(vis_act)
            sys_act = DialogueSystem.act()

            # Visualization
            #print("Visionengine:", vis_utt)
            utts.append("Vision: " + vis_utt)

        sys_utt = sys_act.get("system_utterance", "")
        #print("System:", sys_utt)
        utts.append("System: " + sys_utt)

        turn_image = DialogueSystem.get_image()

        if not (image == turn_image).all():  # changed
            images_and_utterances.append((image, utts))
            image = turn_image
            utts = []

    if not utts:
        utts.append("User: (ends dialogue)")

    images_and_utterances.append((turn_image, utts))

    # Save as figure
    output_png = args.output_png
    create_dialogue_figure(images_and_utterances, output_png)


if __name__ == "__main__":
    main()
