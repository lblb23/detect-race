# Evaluate

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

model = load_model('models/weights.hdf5')

test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

print(model.evaluate_generator(validation_generator, verbose=1))
# best val acc 0.66
