use Hugging Face TFCLIPModel to calculate CLIP score.

* `clip_model.tflite`, `input_ids.bin`, `attention_mask.bin`, and `pixel_values.bin` are generated with the [test_clip_model.ipynb](test_clip_model.ipynb)
* `bazel build --config android_arm64  clip_score` to build android arm64 binary (with `${ANDROID_NDK_HOME}`, `${ANDROID_NDK_VERSION}` and `${ANDROID_NDK_API_LEVEL}` set correctly) 
