import os
import segmentation
import joblib

# load the model
current_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_dir, 'models/svc/svc.pkl')
model = joblib.load(model_dir)

classification_result = []
confidence_threshold = 0.15
for i, each_character in enumerate(segmentation.characters):
    # converts it to a 1D array
    each_character = each_character.reshape(1, -1);

    probas = model.predict_proba(each_character)
    top_preds = sorted(zip(model.classes_, probas[0]), key=lambda x: x[1], reverse=True)[:3]
    best_label, best_confidence = top_preds[0]
    print(f"Caractère #{i+1} — Top prédictions : {top_preds}")
    if best_confidence < confidence_threshold:
        print(f"⚠️ Doute sur le caractère #{i+1} : faible confiance ({best_confidence:.2%})")

    result = model.predict(each_character)
    classification_result.append(result)

print(classification_result)

plate_string = ''
for eachPredict in classification_result:
    plate_string += eachPredict[0]

print("The plate number possibly wrongly arranged : ", plate_string)

# it's possible the characters are wrongly arranged
# since that's a possibility, the column_list will be
# used to sort the letters in the right order

column_list_copy = segmentation.column_list[:]
segmentation.column_list.sort()
rightplate_string = ''
for each in segmentation.column_list:
    rightplate_string += plate_string[column_list_copy.index(each)]

print("This is the plate number : ", rightplate_string)