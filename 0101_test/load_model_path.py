
def load_model(model_name):
    model_dir = 'C:/Users/IVPL-D14/models/research/object_detection/local_models/'+model_name
    #model_dir = 'C:/Users/IVPL-D14/FineTunedModels/'
    model_dir = pathlib.Path(model_dir)/"saved_model"
    print('[INFO] Loading the model from '+ str(model_dir))
    model = tf.saved_model.load(str(model_dir))
    return model


PATH_TO_LABELS = 'C:/Users/IVPL-D14/models/research/object_detection/local_models/knife_label_map.pbtxt'
#PATH_TO_LABELS = 'C:/Users/IVPL-D14/models/research/object_detection/training/labelmap.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)#, use_display_name=True)
model_name = 'trained_model_large_original_15000'
#model_name = '511_batch8_finetuned_model'
print('[INFO] Downloading model and loading to network : '+ model_name)
detection_model = load_model(model_name)
detection_model.signatures['serving_default'].output_dtypes
detection_model.signatures['serving_default'].output_shapes