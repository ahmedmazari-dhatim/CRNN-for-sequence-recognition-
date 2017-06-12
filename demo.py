import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn



#model_path = '/home/ahmed/crnn/data/crnn.pth'
model_path='/home/ahmed/Pictures/model/save_model_first_1/crnn_1.pth'
#img_path = '/home/ahmed/crnn/data/demo.png'
#img_path='/home/ahmed/Pictures/cogedis/2-total.png'
img_path= '/home/ahmed/Pictures/cogedis/cogedis_words_2/2162103c-f6a7-4bed-b2a3-afc2387ca1c2.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'


model = crnn.CRNN(32, 1, 37,100, 1).cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image).cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.squeeze(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
