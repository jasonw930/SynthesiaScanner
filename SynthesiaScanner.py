import cv2
import numpy as np
from tqdm import tqdm
from math import floor, ceil


# TODO: NOTES THAT DON'T APPEAR ON KEYBOARD BECAUSE TOO SHORT
# TODO: key signature changes
# TODO: tie irregular notes
# TODO: clef

# TODO: note padding
# round down -> separate voices -> pad


# User Defined Constants
FILE_NAME = 'AOT'
FPS = 30
BPM = 144
STATE_COLOR = [(190, 252, 81), (89, 196, 209)]
KEYS = [(0, 'C#m'), (23, 'Dm')]

KEY_POS = [(690, 10 + 1257 * i // 51, 1) for i in range(52)]
for i, (r, c, l) in enumerate(KEY_POS[:-1]):
    if i % 7 in [1, 4]:
        continue
    KEY_POS.append((640, c + [17, 0, 11, 15, 0, 9, 13][i % 7], 1))
KEY_POS.sort(key=lambda pos: pos[1])
KEY_POS = list(enumerate(KEY_POS))

# Program Constants
DURATION = ['', '16th', 'eighth', 'eighth',
            'quarter', 'quarter', 'quarter', 'quarter',
            'half', 'half', 'half', 'half',
            'half', 'half', 'half', 'half',
            'whole']
DOTS = [0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 0, 0, 1, 0, 2, 3, 0]
TPC = {
    'C#m': [26, 21, 16, 23, 18, 25, 20, 15, 22, 17, 24, 19],
    'Dm': [14, 21, 16, 11, 18, 13, 8, 15, 10, 17, 12, 19]
}
KEY_SIG = {
    'C#m': 4,
    'Dm': -1
}
LENGTH_MAP = [1, 1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 16]

# # KEY_POS = listof([id, [row, col, delta_col]])
# KEY_POS = [(320, 3 + 1255 * i // 51, 18) for i in range(52)]
# for i, (r, c, l) in enumerate(KEY_POS[:-1]):
#     if i % 7 in [1, 4]:
#         continue
#     KEY_POS.append((320, c + [20, 0, 14, 19, 0, 12, 16][i % 7], 11))
# KEY_POS.sort(key=lambda x: x[1])  # sort by col to assign id
# KEY_POS = list(enumerate(KEY_POS))
# KEY_POS.sort(key=lambda x: x[1][2])  # sort by delta_col to get black keys first
# # Scan black, destroy, scan white


key_state = [[] for i in range(88)]
vidcap = cv2.VideoCapture(FILE_NAME + '.mp4')
read_frames = 0
total_note_frames = 0
with tqdm() as pbar:
    while True:
        success, image = vidcap.read()
        read_frames += 1
        if not success or read_frames > FPS * 1000: break
        if read_frames < FPS: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i, (r, c, l) in KEY_POS:
            dists = [sum(np.square(image[r, c] - sc)) for sc in STATE_COLOR]
            min_val, min_i = min([(b, a) for a, b in enumerate(dists)])
            if min_val < 10000:
                key_state[i].append(min_i + 1)
                total_note_frames += 1
            else:
                key_state[i].append(0)

            # ALTERNATIVE SCANNING to deal with notes that are too short, kinda doesn't work
            # min_val = 99999999
            # min_state = 0
            # delta_rows = [0, 2, 4]
            # for state_num, scol in enumerate(STATE_COLOR):
            #     for dr in delta_rows:
            #         dists = [sum(np.square(image[r+dr, c+dc] - scol)) for dc in range(0, l, 2)]
            #         if sum(np.array(dists) < 20000) < len(dists) * 0.9: continue
            #         min_val, min_state = min([(min_val, min_state), (sum(dists)/len(dists), state_num+1)])
            #         break
            # if min_state != 0:
            #     for dr in delta_rows:
            #         image[[r+dr]*l, range(c, c+l)] = [0, 0, 0]
            #     total_note_frames += 1
            # key_state[i].append(min_state)

        # if read_frames == 200:
        #     for i, (r, c, l) in enumerate(KEY_POS):
        #         image[[r]*l, range(c, c+l)] = [255, 0, 0]
        #     cv2.imwrite(FILE_NAME + '_highlight_2.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        pbar.update(1)

print('%d note frames read' % total_note_frames)
key_state = np.array(key_state)

key_state_rle = []
for index in range(88):
    ia = key_state[index]
    n = len(ia)
    y = ia[1:] != ia[:-1]
    i = np.append(np.where(y), n - 1)
    z = np.diff(np.append(-1, i))
    p = np.cumsum(np.append(0, z))[:-1]
    key_state_rle.append(np.transpose([z, p, ia[i]]))
key_state_rle = [l.tolist() for l in key_state_rle]
# key_state_rle[key, ith-note] = [length, pos, state]
print('%d notes read' % sum([len(x) for x in key_state_rle]))

start_frame = min([l[0][0] for l in key_state_rle])


def skew_round(x):
    if x % 1 < 0.15 / 10:
        return floor(x)
    return ceil(x)


measures = [[[] for j in range(200)] for i in range(len(STATE_COLOR))]
# measures[state, m_num] = listof([s_num, length, key, tie])
for key in range(88):
    for frame_len, frame_pos, state in key_state_rle[key]:
        if state > 0:
            pos = int(round((frame_pos - start_frame) / FPS / 60 * BPM * 4))
            length = int(skew_round((frame_pos + frame_len - start_frame) / FPS / 60 * BPM * 4)) - pos
            if length <= 0: print('bruh')

            m_num = pos // 16
            s_num = pos % 16

            queued_tie = []
            while s_num + length > 16:
                measures[state-1][m_num].append([s_num, 16 - s_num, key, queued_tie + [16 - s_num]])
                queued_tie = [-(16 - s_num)]
                length -= 16 - s_num
                s_num = 0
                m_num += 1

                if length <= 0: print('big bruh')
            measures[state-1][m_num].append([s_num, length, key, queued_tie])

# for i in measures[0][:4]:
#     print(i)
# print()
# for i in measures[1][:4]:
#     print(i)


# for i in measures[0][62]:
#     print(i)
# print()
# for i in measures[0][63]:
#     print(i)

# LENGTH ADJUSTMENT: map and tie
for s in range(len(STATE_COLOR)):
    for m_num in range(200):
        for i in range(len(measures[s][m_num])):
            measures[s][m_num][i][1] = LENGTH_MAP[measures[s][m_num][i][1]]
            measures[s][m_num][i][1] = max([measures[s][m_num][i][1]] + measures[s][m_num][i][3])

# CHORDIFY
# measures[state, m_num] = listof([s_num, length, key, tie])
# measures[state, m_num] = listof([chord_s_num, chord_len, listof([key, tie])])
for state in range(len(STATE_COLOR)):
    for m_num in range(200):
        measures[state][m_num].sort(key=lambda x: x[0])
        chords = []
        for s_num, length, key, tie in measures[state][m_num]:
            for chord_s_num, chord_len, chord_notes in chords:
                if s_num == chord_s_num and length == chord_len:
                    chord_notes.append([key, tie])
                    break
            else:
                chords.append([s_num, length, [[key, tie]]])
        measures[state][m_num] = chords
# Remove middle of 3 semitone clusters
# for state in range(len(STATE_COLOR)):
#     for m_num in range(200):
#         for i, (chor_s_num, chord_len, chord_notes) in enumerate(measures[state][m_num]):
#             pruned_notes = []
#             keys = [note[0] for note in chord_notes]
#             for note in chord_notes:
#                 if note[0]-1 not in keys or note[0]+1 not in keys:
#                     pruned_notes.append(note)
#             measures[state][m_num][i][2] = pruned_notes


# for i in measures[0][:4]:
#     print(i)
# print()
# for i in measures[1][:4]:
#     print(i)

# VOICEIFY
# measures[state, m_num, chord] = [chord_s_num, chord_len, listof([key, tie])]
# measures[state, m_num, voice, chord] = [chord_s_num, chord_len, listof([key, tie])]
for state in range(len(STATE_COLOR)):
    for m_num in range(200):
        measures[state][m_num].sort(key=lambda x: (x[0], -(max(x[2])[0])))
        voices = []
        while len(measures[state][m_num]) > 0:
            voice = [measures[state][m_num][0]]
            remaining_chords = []
            for chord_s_num, chord_len, chord_notes in measures[state][m_num][1:]:
                if chord_s_num >= voice[-1][0] + voice[-1][1]:
                    voice.append([chord_s_num, chord_len, chord_notes])
                else:
                    remaining_chords.append([chord_s_num, chord_len, chord_notes])
            measures[state][m_num] = remaining_chords
            voices.append(voice)
        measures[state][m_num] = voices

for state in range(len(STATE_COLOR)):
    for m_num in range(200):
        if len(measures[state][m_num]) == 0:
            measures[state][m_num] = [[]]
# for i in measures[0][:4]:
#     print(i)
# print()
# for i in measures[1][:4]:
#     print(i)


# LENGTH ADJUSTMENT: pad
for state in range(len(STATE_COLOR)):
    for m_num in range(200):
        for voice_num, voice in enumerate(measures[state][m_num]):
            for i in range(len(voice) - 1):
                voice[i][1] = min(voice[i][1] * 4, voice[i+1][0] - voice[i][0])
            if len(voice) > 0:
                voice[-1][1] = min(voice[-1][1] * 4, 16 - voice[-1][0])


# XML
# Just in time irregular ties
output = []
notes_written = 0


def rest_xml(start, end):
    result = []
    if start == 0 and end == 16:
        result.append('<Rest>')
        result.append('<durationType>measure</durationType>')
        result.append('<duration>4/4</duration>')
        result.append('</Rest>')
        return result
    while start != 0 and start + (start & -start) <= end:
        result.append('<Rest>')
        result.append('<durationType>%s</durationType>' % DURATION[start & -start])
        result.append('</Rest>')
        start += (start & -start)  # LSB
    end -= start
    while end > 0:
        msb = end
        msb = msb | (msb >> 1)
        msb = msb | (msb >> 2)
        msb = msb | (msb >> 4)
        msb = (msb + 1) >> 1
        result.append('<Rest>')
        result.append('<durationType>%s</durationType>' % DURATION[msb])
        result.append('</Rest>')
        end -= msb
    return result


for state in range(len(STATE_COLOR)):
    output.append('<Staff id="%d">' % (state + 1))
    if state == 0:
        output.append(('<VBox>'
                       '<height>10</height>'
                       '<Text>'
                       '<style>Title</style>'
                       '<text>%s</text>'
                       '</Text>'
                       '</VBox>') % FILE_NAME)
    keys_copy = KEYS.copy()
    cur_key = (0, 'CM')
    for m_num in range(200):  # tqdm(range(200)):
        while len(keys_copy) > 0 and m_num == keys_copy[0][0]:
            cur_key = keys_copy[0]
            del keys_copy[0]
        output.append('<Measure>')
        for voice_num, voice in enumerate(measures[state][m_num]):
            output.append('<voice>')
            if m_num == cur_key[0] and voice_num == 0:
                output.append(('<KeySig>'
                               '<accidental>%d</accidental>'
                               '</KeySig>') % KEY_SIG[cur_key[1]])
                output.append('<TimeSig>'
                              '<sigN>4</sigN>'
                              '<sigD>4</sigD>'
                              '</TimeSig>')
            if m_num == 0 and voice_num == 0 and state == 0:
                output.append(('<Tempo>'
                               '<tempo>%f</tempo>'
                               '<followText>1</followText>'
                               '<text><b></b><font face="ScoreText"/><b><font face="FreeSerif"/> = %d</b></text>'
                               '</Tempo>') % (BPM / 60, BPM))
            cur_s_pos = 0
            for chord_s_num, chord_len, chord_notes in voice:
                if cur_s_pos < chord_s_num:
                    output += rest_xml(cur_s_pos, chord_s_num)
                    cur_s_pos = chord_s_num
                output.append('<Chord>')
                if DOTS[chord_len] > 0:
                    output.append('<dots>%d</dots>' % DOTS[chord_len])
                output.append('<durationType>%s</durationType>' % DURATION[chord_len])
                for key, tie in chord_notes:
                    output.append('<Note>')
                    for t in tie:
                        if t > 0:
                            output.append('<Spanner type="Tie">'
                                          '<Tie>'
                                          '</Tie>'
                                          '<next>'
                                          '<location>'
                                          '<measures>1</measures>')
                            if t < 16: output.append('<fractions>%d/16</fractions>' % (t - 16))
                            output.append('</location>'
                                          '</next>'
                                          '</Spanner>')
                        elif t < 0:
                            output.append('<Spanner type="Tie">'
                                          '<prev>'
                                          '<location>'
                                          '<measures>-1</measures>')
                            if t > -16: output.append('<fractions>%d/16</fractions>' % (t + 16))
                            output.append('</location>'
                                          '</prev>'
                                          '</Spanner>')
                    output.append('<pitch>%d</pitch>' % (key + 21))
                    output.append('<tpc>%d</tpc>' % TPC[cur_key[1]][(key + 21) % 12])
                    output.append('</Note>')
                    notes_written += 1
                output.append('</Chord>')
                cur_s_pos += chord_len
            if cur_s_pos < 16:
                output += rest_xml(cur_s_pos, 16)
            output.append('</voice>')
        output.append('</Measure>')
    output.append('</Staff>')
print('%d notes written' % notes_written)

with open(FILE_NAME + '.mscx', 'w') as file:
    with open('prefix.txt', 'r') as prefix:
        file.write(prefix.read())
    file.write('\n'.join(output))
    with open('postfix.txt', 'r') as postfix:
        file.write(postfix.read())


# image = cv2.cvtColor(cv2.imread('AOT_frame.png'), cv2.COLOR_BGR2RGB)
# for x, y in KEY_POS:
#     image[[x, x, x+1, x+1], [y, y+1, y, y+1]] = [255, 0, 0]
# cv2.imwrite('AOT_highlight.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
