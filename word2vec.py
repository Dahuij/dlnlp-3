import os
import re
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.feature_extraction.text import TfidfVectorizer


try:
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except:
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False

# 设置路径
base_path = r"C:\Users\wjl\Documents\dlnlp\dlnlp-3"
corpus_dir = os.path.join(base_path, "jyxstxtqj_downcc.com")  # 金庸小说文本所在目录

# --------------------------------------------------
# 1. 数据预处理与语料加载
# --------------------------------------------------
def load_corpus_from_folder(folder_path):
    all_sentences = []
    # 添加重要词汇
    jieba.suggest_freq('降龙十八掌', tune=True)
    jieba.suggest_freq('打狗棒法', tune=True)
    jieba.suggest_freq('郭靖', tune=True)
    jieba.suggest_freq('杨过', tune=True)
    jieba.suggest_freq('小龙女', tune=True)
    jieba.suggest_freq('金轮法王', tune=True)
    jieba.suggest_freq('黯然销魂掌', tune=True)
    
    # 只读取神雕侠侣文件
    target_file = "神雕侠侣.txt"
    filepath = os.path.join(folder_path, target_file)
    
    if not os.path.exists(filepath):
        print(f"错误：找不到文件 {target_file}")
        return all_sentences
        
    # 尝试的编码列表
    encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8', 'big5']
    
    # 尝试不同的编码方式读取文件
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
                print(f"使用 {encoding} 编码成功读取文件")
                text = f.read().replace('\n', '')  # 去除换行符
                raw_sentences = re.split(r'[。！？]', text)  # 分句
                for sent in raw_sentences:
                    sent = sent.strip()
                    if sent:
                        # 分词 + 过滤单字和标点
                        words = jieba.lcut(sent)
                        words = [w for w in words if len(w) > 1 and not re.match(r'^\W+$', w)]
                        if words:  # 确保句子不为空
                            all_sentences.append(words)
                break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
            continue
            
    print(f"成功加载 {len(all_sentences)} 个句子")
    return all_sentences

# --------------------------------------------------
# 2. 训练Word2Vec模型
# --------------------------------------------------
def train_word2vec(sentences):
    if not sentences:
        print("错误：没有可用的语料")
        return None
    
    print(f"开始训练，语料大小：{len(sentences)}句")
    model = Word2Vec(
        sentences=sentences,
        vector_size=200,   # 词向量维度
        window=10,          # 上下文窗口大小
        min_count=5,        # 忽略出现次数低于5的词
        workers=4,          # 并行线程数
        epochs=10,          # 训练轮次
        sg=1                # 使用Skip-gram模型
    )
    
    # 保存模型
    model.save(os.path.join(base_path, "word2vec_shediao.model"))
    print(f"模型训练完成，词表大小：{len(model.wv.key_to_index)}个词")
    return model

# --------------------------------------------------
# 3. 显示与指定词最相似的词
# --------------------------------------------------
def show_similar_words(model, keyword, topn=10):
    print(f"\n与「{keyword}」最相似的词：")
    try:
        similar = model.wv.most_similar(keyword, topn=topn)
        for word, score in similar:
            print(f"{word}: {score:.4f}")
    except KeyError:
        print(f"词语「{keyword}」不在词表中")

# --------------------------------------------------
# 4. 类比推理：A 之于 B，如同 C 之于 ?
# --------------------------------------------------
def analogy_test(model, word_a, word_b, word_c):
    print(f"\n「{word_a}」之于「{word_b}」，如同「{word_c}」之于：")
    try:
        result = model.wv.most_similar(positive=[word_c, word_b], negative=[word_a])
        for word, score in result[:5]:
            print(f"{word}: {score:.4f}")
    except KeyError as e:
        print(f"词语错误：{e}")

# --------------------------------------------------
# 5. 词向量聚类并可视化
# --------------------------------------------------
def plot_words(model, words):
    vectors = []
    valid_words = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
            valid_words.append(word)
        else:
            print(f"警告：词语「{word}」不在词表中，将被跳过")

    if len(vectors) < 2:
        print("可用的词向量太少，无法可视化。")
        return

    vectors = np.array(vectors)

    # 使用TSNE降维可视化
    tsne = TSNE(n_components=2, perplexity=min(30, len(vectors) - 1), random_state=42)
    reduced = tsne.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(valid_words):
        plt.scatter(reduced[i, 0], reduced[i, 1], c='blue', s=80)
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=12)
    
    plt.title("神雕侠侣人物词向量聚类")
    plt.tight_layout()
    plt.savefig('word_vectors_5.png', dpi=300, bbox_inches='tight')
    print("\n词向量可视化已保存为 'word_vectors_5.png'")
    plt.close()
    
    # 额外添加词向量聚类分析
    if len(valid_words) >= 3:
        # K-Means聚类
        n_clusters = min(3, len(valid_words))
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(vectors)
        
        # PCA降维可视化
        pca = PCA(n_components=2)
        points = pca.fit_transform(vectors)
        
        plt.figure(figsize=(10, 8))
        for i, word in enumerate(valid_words):
            plt.scatter(points[i, 0], points[i, 1], c=clusters[i], cmap='viridis', s=100)
            plt.text(points[i, 0]+0.02, points[i, 1]+0.02, word, fontsize=12)
        plt.title("神雕侠侣人物聚类分析 (K-means)")
        plt.tight_layout()
        plt.savefig('character_clusters_5.png', dpi=300, bbox_inches='tight')
        print("聚类分析结果已保存为 'character_clusters_5.png'")
        plt.close()

# --------------------------------------------------
# 主程序
# --------------------------------------------------
if __name__ == "__main__":
    print("加载神雕侠侣语料中...")
    corpus = load_corpus_from_folder(corpus_dir)

    if not corpus:
        print("错误：未能成功加载语料")
        exit()

    print(f"共加载 {len(corpus)} 个句子，开始训练 Word2Vec...")
    model = train_word2vec(corpus)
    
    if not model:
        print("错误：模型训练失败")
        exit()
        
    print("训练完成！")

    # 计算TF-IDF权重用于后续分析
    corpus_text = [' '.join(words) for words in corpus]
    tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(), lowercase=False)
    tfidf.fit(corpus_text)

    # === 相似词演示 ===
    show_similar_words(model, '杨过')
    show_similar_words(model, '小龙女')
    show_similar_words(model, '郭襄')

    # === 类比推理演示 ===
    analogy_test(model, '杨过', '小龙女', '郭靖')  # 杨过:小龙女 = 郭靖:?
    analogy_test(model, '郭靖', '降龙十八掌', '杨过')  # 郭靖:降龙十八掌 = 杨过:?

    # === 聚类可视化 ===
    characters = ['杨过', '小龙女', '郭襄', '郭芙', '郭靖', '黄蓉',
                  '金轮法王', '李莫愁', '陆无双', '程英', '耶律齐',
                  '完颜萍']
    plot_words(model, characters)
