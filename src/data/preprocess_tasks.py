import sys
import random
import antlr4
from antlr4.InputStream import InputStream
from data.antlr_parsers.java.Java8Lexer import Java8Lexer
from data.antlr_parsers.java.Java8Parser import Java8Parser
from generator_network import generator_java


def get_java_tokens(code):
    lexer = Java8Lexer(InputStream(code))
    stream = antlr4.CommonTokenStream(lexer)
    parser = Java8Parser(stream)
    tree = parser.compilationUnit()

    token_types = []
    for token in stream.tokens:
        if token.type == Java8Parser.Identifier:
            token_types.append('1')
        else:
            token_types.append('0')

    return ' '.join(token_types)


def preprocess_it(code):
    tags = get_java_tokens(code)

    return code, tags


def preprocess_rtd(code):
    code_tokens = code.split()
    num_masked_tokens = int(0.15 * len(code_tokens))
    masked_ids = sorted([random.randrange(len(code_tokens)) for x in range(num_masked_tokens)])

    new_code = []
    outputs = []
    batch_to_generate = []
    for i in range(len(code_tokens)):
        if (i in masked_ids) and (i > 0):
            context = ' '.join(code_tokens[(i - 100 if (i - 100) >= 0 else 0):i])
            batch_to_generate.append((context, i))
            new_code.append('[MASK]')
            outputs.append('[MASK]')
        elif (i in masked_ids) and (i == 0):
            java_key_words = [
                "abstract", "boolean", "byte", "break", "class", "case", "catch", "char",
                "continue", "default", "do", "double", "else", "extends", "final", "finally",
                "float", "for", "if", "implements", "import", "instanceof", "int", "interface",
                "long", "native", "new", "package", "private", "protected", "public", "return",
                "short", "static", "super", "switch", "synchronized", "this", "throw", "throws",
                "transient", "try", "void", "volatile", "while", "assert", "const", "enum", "goto", 
                "strictfp"
            ]
            random_token = random.choice(java_key_words)
            new_code.append(random_token)
            if random_token == code_tokens[i]:
                # original
                outputs.append('1')
            else:
                # replaced
                outputs.append('0')
        else:
            # original
            new_code.append(code_tokens[i])
            outputs.append('1')
    
    gen_tokens = generator_java.generate_next_token([x[0] for x in batch_to_generate], False)
    print(gen_tokens)
    for token, sample in zip(gen_tokens, batch_to_generate):
        new_code[sample[1]] = token
        if token == code_tokens[sample[1]]:
            # original
            outputs[sample[1]] = '1'
        else:
            # replaced
            outputs[sample[1]] = '0'

    return  ' '.join(new_code), ' '.join(outputs)
