python eval_net.py -g 0 \
	-p ./models/deploy_inceptionv2_8_ave.prototxt \
    -m ./models/pretrained.caffemodel \
	-i eval_additional_illegal_passanger_label.txt \
	-o ./evalresult/ \
	--scale 0.0078125 --mean 117 117 117 \
    --batch_size 24 \
	--result_layer softmax_orientation softmax_gender softmax_glass softmax_hat softmax_helmet softmax_mask softmax_nation softmax_backpack softmax_shoulderbag softmax_transportationkind softmax_roof softmax_seal softmax_upperpattern softmax_raincoat softmax_transportationcolor softmax_uppercolor softmax_illegalpassenger
python eval_multi_score.py ./evalresult/scorefile.npz --attrnamefile label.list
