import argparse
import pickle
from tqdm import tqdm
from datasets.baseline_dataloader import CodeRow


def extract(lmdb_env, loader, model, device):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)

        for img, class_id, salience, filename in pbar:
            img = img.to(device)

            _, _, id_b = model.encode(img)
            # id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()

            for c_id, sali, file, bottom in zip(class_id, salience, filename, id_b):
                row = CodeRow(
                    bottom=bottom, class_id=c_id, salience=sali, filename=file
                )
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))



