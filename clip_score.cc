#include <fstream>
#include <iostream>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

using namespace std;

int main(int argc, char *argv[]) {
  std::vector<int32_t> input_ids(77);
  std::vector<int32_t> attention_mask(77);
  std::vector<float> pixel_values(3 * 224 * 224);
  std::ifstream input_ids_file("input_ids.bin",
                               std::ios::in | std::ios::binary);
  std::ifstream attention_mask_file("attention_mask.bin",
                                    std::ios::in | std::ios::binary);
  std::ifstream pixel_values_file("pixel_values.bin",
                                  std::ios::in | std::ios::binary);

  input_ids_file.read((char *)input_ids.data(), 77 * sizeof(int32_t));
  attention_mask_file.read((char *)attention_mask.data(), 77 * sizeof(int32_t));
  pixel_values_file.read((char *)pixel_values.data(),
                         (3 * 224 * 224) * sizeof(float));

  auto model = tflite::FlatBufferModel::BuildFromFile("clip_model.tflite");
  if (model == nullptr) {
    cout << "failed to load model "
         << "clip_model.tflite\n";
    exit(-1);
  }

  std::unique_ptr<tflite::Interpreter> interpreter;

  tflite::ops::builtin::BuiltinOpResolver resolver;
  if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
    cout << "failed to build interpreter\n";
    exit(-1);
  }
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    cout << "failed allocate tensors\n";
    exit(-1);
  }

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  std::copy(attention_mask.begin(), attention_mask.end(),
            interpreter->typed_input_tensor<int32_t>(0));
  cout << endl;
  std::copy(input_ids.begin(), input_ids.end(),
            interpreter->typed_input_tensor<int32_t>(1));
  std::copy(pixel_values.begin(), pixel_values.end(),
            interpreter->typed_input_tensor<float>(2));

  if (interpreter->Invoke() != kTfLiteOk) {
    cout << "Failed to invoke tflite!\n";
    exit(-1);
  }
  auto output_1 = interpreter->typed_tensor<float>(outputs[1]);
  auto output_2 = interpreter->typed_tensor<float>(outputs[2]);
  std::vector<float> o_1(output_1,
                         output_1 + interpreter->tensor(outputs[1])->bytes / 4);
  std::vector<float> o_2(output_2,
                         output_2 + interpreter->tensor(outputs[2])->bytes / 4);

  cout << "output: " << o_1[0] << std::endl;
  cout << "output: " << o_2[0] << std::endl;
}
