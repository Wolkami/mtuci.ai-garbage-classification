from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split


RAW_DIR = Path("data/raw/recyclable-and-household-waste-classification")
OUT_DIR = Path("data/processed")

SEED = 42

CLASS_MAPPING = {
    # metal
    "aerosol_cans": "metal",
    "aluminum_food_cans": "metal",
    "aluminum_soda_cans": "metal",
    "steel_food_cans": "metal",

    # cardboard
    "cardboard_boxes": "cardboard",
    "cardboard_packaging": "cardboard",

    # paper
    "magazines": "paper",
    "newspaper": "paper",
    "office_paper": "paper",
    "paper_cups": "paper",

    # glass
    "glass_beverage_bottles": "glass",
    "glass_cosmetic_containers": "glass",
    "glass_food_jars": "glass",

    # plastic
    "disposable_plastic_cutlery": "plastic",
    "plastic_cup_lids": "plastic",
    "plastic_detergent_bottles": "plastic",
    "plastic_food_containers": "plastic",
    "plastic_shopping_bags": "plastic",
    "plastic_soda_bottles": "plastic",
    "plastic_straws": "plastic",
    "plastic_trash_bags": "plastic",
    "plastic_water_bottles": "plastic",

    # other
    "clothing": "other",
    "coffee_grounds": "other",
    "eggshells": "other",
    "food_waste": "other",
    "shoes": "other",
    "styrofoam_cups": "other",
    "styrofoam_food_containers": "other",
    "tea_bags": "other",
}


def collect_images():
    items = []

    for source_class, target_class in CLASS_MAPPING.items():
        class_dir = RAW_DIR / source_class

        for domain in ["default", "real_world"]:
            domain_dir = class_dir / domain

            if not domain_dir.exists():
                print(f"Skip missing folder: {domain_dir}")
                continue

            for image_path in domain_dir.rglob("*"):
                if image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    items.append({
                        "path": image_path,
                        "source_class": source_class,
                        "target_class": target_class,
                        "domain": domain,
                    })

    return items


def copy_items(items, split_name):
    for item in items:
        target_dir = OUT_DIR / split_name / item["target_class"]
        target_dir.mkdir(parents=True, exist_ok=True)

        new_name = f"{item['source_class']}__{item['domain']}__{item['path'].name}"
        shutil.copy2(item["path"], target_dir / new_name)


def main():
    items = collect_images()

    print(f"Total images: {len(items)}")

    labels = [f"{item['target_class']}__{item['domain']}" for item in items]

    train_items, temp_items = train_test_split(
        items,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )

    temp_labels = [f"{item['target_class']}__{item['domain']}" for item in temp_items]

    val_items, test_items = train_test_split(
        temp_items,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_labels,
    )

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    copy_items(train_items, "train")
    copy_items(val_items, "val")
    copy_items(test_items, "test")

    print(f"Train: {len(train_items)}")
    print(f"Val:   {len(val_items)}")
    print(f"Test:  {len(test_items)}")
    print("Done.")


if __name__ == "__main__":
    main()