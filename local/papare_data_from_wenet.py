import os
import sys
import multiprocessing
from typing import List
from datasets import Dataset
import copy


INIT_ITEM = {'audio': '',
             'id': '',
             'sentence': ''}


def read_file(path: str) -> List:
    with open(path, 'r') as f:
        file = f.readlines()
    return [item.strip().split(' ') for item in file]


def list2dict(wav_list: List, text_list: List, start: int, end: int, output_queue) -> None:
    local_labeled_data_list = []
    for i in range(start, end):
        item = copy.deepcopy(INIT_ITEM)
        if len(text_list[i]) <= 1:
            continue
        path = wav_list[i][1]
        sentence = text_list[i][1]

        item['audio'] = path
        item['sentence'] = sentence
        item['id'] = wav_list[i][0]
        local_labeled_data_list.append(item)

    output_queue.put(local_labeled_data_list)


def merge_generate_json(path: List, export_dir: str) -> None:
    num_processes = multiprocessing.cpu_count()
    output_queue = multiprocessing.Queue()

    processes = []
    for split in path:
        wav_list = read_file(os.path.join(split, 'wav.scp'))
        text_list = read_file(os.path.join(split, 'text'))
        length = len(wav_list)
        chunk_size = length // num_processes

        for i in range(num_processes):
            start = i * chunk_size
            end = length if i == num_processes - 1 else (i + 1) * chunk_size
            process = multiprocessing.Process(target=list2dict, args=(wav_list, text_list, start, end, output_queue))
            processes.append(process)
            process.start()

    labeled_data_list = []
    for _ in range(num_processes):
        labeled_data_list.extend(output_queue.get())

    for p in processes:
        p.join()

    print('Writing data ...')
    dataset = Dataset.from_list(labeled_data_list)
    dataset.save_to_disk(export_dir)
    # with open(export_path, 'w') as f:
    #     for item in tqdm(labeled_data_list):
    #         data = json.dumps(item)
    #         f.write(data + '\n')


if __name__ == "__main__":
    wenet_dir = sys.argv[1]
    save_dir = sys.argv[2]

    path_list = [wenet_dir]
    export_root_path = save_dir
    os.makedirs(export_root_path, exist_ok=True)

    merge_generate_json(path_list, save_dir)
