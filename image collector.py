import os
import shutil
import random

def copy_random_photos(source_folder, destination_folder, num_files=400):
    # 1. Έλεγχος αν υπάρχει ο φάκελος προέλευσης
    if not os.path.exists(source_folder):
        print(f"Σφάλμα: Ο φάκελος '{source_folder}' δεν βρέθηκε.")
        return

    # 2. Δημιουργία του φακέλου προορισμού αν δεν υπάρχει
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Δημιουργήθηκε ο φάκελος: {destination_folder}")

    # 3. Λήψη λίστας όλων των αρχείων από τον φάκελο (φιλτράρισμα μόνο για εικόνες)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    all_photos = [f for f in os.listdir(source_folder) 
                  if f.lower().endswith(valid_extensions)]

    # 4. Έλεγχος αν υπάρχουν αρκετές φωτογραφίες
    if len(all_photos) < num_files:
        print(f"Προειδοποίηση: Βρέθηκαν μόνο {len(all_photos)} φωτογραφίες. Θα αντιγραφούν όλες.")
        num_files = len(all_photos)

    # 5. Τυχαία επιλογή αρχείων
    selected_photos = random.sample(all_photos, num_files)

    # 6. Αντιγραφή των αρχείων
    print(f"Έναρξη αντιγραφής {num_files} αρχείων...")
    for file_name in selected_photos:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.copy2(source_path, destination_path) # Το copy2 διατηρεί και τα metadata

    print("Η αντιγραφή ολοκληρώθηκε επιτυχώς!")

# --- ΡΥΘΜΙΣΕΙΣ ---
source = r"C:\Users\arsen\Downloads\archive\grape_24k_fixed\Leaf Blight"
destination = r"C:\Users\arsen\OneDrive\Desktop\leaf blight 400" # Βάλε εδώ τη διαδρομή του νέου φακέλου
amount = 400                            # Πόσες φωτογραφίες θέλεις

# Εκτέλεση
copy_random_photos(source, destination, amount)