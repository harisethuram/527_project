# preprocess_chexpert_fast_u8.py
import argparse, os, numpy as np
from datasets import load_dataset
from torchvision import transforms

TARGET_LABELS = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Pleural Effusion"]

def preprocess_batch(batch, resolution):
    resize = transforms.Resize((resolution, resolution))

    images = []
    labels = []

    for img, *vals in zip(batch["image"], *[batch[k] if k in batch else [3]*len(batch["image"]) for k in TARGET_LABELS]):
        img = img.convert("RGB")
        img = resize(img)
        arr = np.asarray(img, dtype=np.uint8)          # HWC
        arr = np.transpose(arr, (2, 0, 1))             # CHW
        images.append(arr)

        labs = []
        for v in vals:
            if v == 1:
                labs.append(1.0)
            elif v == 3:
                labs.append(-1.0)
            else:
                labs.append(0.0)
        labels.append(np.asarray(labs, dtype=np.float32))

    return {"image_u8": images, "labels": labels}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=128)  # batching for map()
    args = p.parse_args()
    print(args)
    os.makedirs(args.out_dir, exist_ok=True)
    print("Loading dataset...")
    ds = load_dataset("danjacobellis/chexpert")  # train/validation
    
    for split in ds.keys():
        print(f"Split {split} n={len(ds[split])}")

        processed = ds[split].map(
            lambda b: preprocess_batch(b, args.resolution),
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            remove_columns=ds[split].column_names,
            desc=f"preprocess {split}",
        )

        split_out = os.path.join(args.out_dir, split)
        processed.save_to_disk(split_out)
        print(f"Saved {split_out}")

if __name__ == "__main__":
    main()