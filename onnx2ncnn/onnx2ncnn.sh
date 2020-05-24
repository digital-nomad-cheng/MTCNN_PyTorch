python -m onnxsim pnet.onnx pnet_sim.onnx
python -m onnxsim rnet.onnx rnet_sim.onnx
python -m onnxsim onet.onnx onet_sim.onnx
echo "Finished simplified"
~/Libs/ncnn/build/tools/onnx/onnx2ncnn pnet_sim.onnx mobile_pnet_sim.param mobile_pnet_sim.bin
~/Libs/ncnn/build/tools/onnx/onnx2ncnn rnet_sim.onnx mobile_rnet_sim.param mobile_rnet_sim.bin
~/Libs/ncnn/build/tools/onnx/onnx2ncnn onet_sim.onnx mobile_onet_sim.param mobile_onet_sim.bin
