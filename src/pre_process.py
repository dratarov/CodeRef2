import sys
import re
import random
import pandas as pd

from data.vocab import load_vocab

df = pd.read_csv('../dataset/fine-tuning/new_large/code&comment-to-code/train.tsv', sep='\t', header=None)
# df_test = pd.read_csv('../dataset/fine-tuning/new_large/code&comment-to-code/test.tsv', sep='\t', header=None)
# df_val = pd.read_csv('../dataset/fine-tuning/new_large/code&comment-to-code/val.tsv', sep='\t', header=None)

# code_vocab = load_vocab(vocab_root='../SPT-Code-Dataset/dataset/vocab_saved_new', name='code')
# nl_vocab = load_vocab(vocab_root='../SPT-Code-Dataset/dataset/vocab_saved_new', name='nl')

df = pd.concat([df_train, df_test, df_val])
j = 0
	# test_buggy = open("../SPT-Code-Dataset/dataset/fine_tune/bug_fix2/small/test.buggy-fixed.buggy",'w')
	# test_fixed = open("../SPT-Code-Dataset/dataset/fine_tune/bug_fix2/small/test.buggy-fixed.fixed",'w')

	# train_buggy = open("../SPT-Code-Dataset/dataset/fine_tune/bug_fix2/small/train.buggy-fixed.buggy",'w')
	# train_fixed = open("../SPT-Code-Dataset/dataset/fine_tune/bug_fix2/small/train.buggy-fixed.fixed",'w')

	# valid_buggy = open("../SPT-Code-Dataset/dataset/fine_tune/bug_fix2/small/valid.buggy-fixed.buggy",'w')
# valid_fixed = open("../SPT-Code-Dataset/dataset/fine_tune/bug_fix2/small/valid.buggy-fixed.fixed",'w')
l = 0
for i in range(len(df)):
	pair = df.iloc[i]

	if comment_length > l:
		l = comment_length
	rand_num = random.random()
	if rand_num <= 0.1:
		test_buggy.write(pair[0] + '\n')
		test_fixed.write(pair[1] + '\n')
	elif rand_num > 0.1 and rand_num <= 0.2:
		valid_buggy.write(pair[0] + '\n')
		valid_fixed.write(pair[1] + '\n')
	elif rand_num > 0.2:
		train_buggy.write(pair[0] + '\n')
		train_fixed.write(pair[1] + '\n')
	j +=1

	print(i)


print(j)
print(l)
test_buggy.close()
test_fixed.close()

train_buggy.close()
train_fixed.close()

valid_buggy.close()
valid_fixed.close()
