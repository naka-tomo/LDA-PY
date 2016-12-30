# encoding: shift_jis
import numpy
import random
import math
import matplotlib
import pylab
import matplotlib.pyplot as plt

# ハイパーパラメータ
__alpha = 1.0
__beta = 1.0


def calc_lda_param( docs_dn, topics_dn, K, V ):
    D = len(docs_dn)

    n_dz = numpy.zeros((D,K))   # 各文書dにおいてトピックzが発生した回数
    n_zv = numpy.zeros((K,V))   # 各トピックzにぴおいて単語vが発生した回数
    n_z = numpy.zeros(K)        # 各トピックが発生した回数

    # 数え上げる
    for d in range(D):
        N = len(docs_dn[d])    # 文書に含まれる単語数
        for n in range(N):
            v = docs_dn[d][n]  # ドキュメントdのn番目の単語のインデックス
            z = topics_dn[d][n]     # 単語に割り当てれれているトピック
            n_dz[d][z] += 1
            n_zv[z][v] += 1
            n_z[z] += 1

    return n_dz, n_zv, n_z


def sample_topic( d, v, n_dz, n_zv, n_z, K, V ):
    P = [ 0.0 ] * K

    # 累積確率を計算
    P = (n_dz[d,:] + __alpha )*(n_zv[:,v] + __beta) / (n_z[:] + V *__beta)
    for z in range(1,K):
        P[z] = P[z] + P[z-1]

    # サンプリング
    rnd = P[K-1] * random.random()
    for z in range(K):
        if P[z] >= rnd:
            return z



# 単語を一列に並べたリスト変換
def conv_to_word_list( data ):
    V = len(data)
    doc = []
    for v in range(V):  # v:語彙のインデックス
        for n in range(data[v]): # 語彙の発生した回数文forを回す
            doc.append(v)
    return doc

# 尤度計算
def calc_liklihood( data, n_dz, n_zv, n_z, K, V  ):
    lik = 0

    # 上の処理を高速化
    P_vz = (n_zv.T + __beta) / (n_z + V *__beta)
    for d in range(len(data)):
        Pz = (n_dz[d] + __alpha )/( numpy.sum(n_dz[d]) + K *__alpha )
        Pvz = Pz * P_vz
        Pv = numpy.sum( Pvz , 1 ) + 0.000001
        lik += numpy.sum( data[d] * numpy.log(Pv) )

    return lik

def save_model( n_dz, n_zv, n_z ):
    Pdz = n_dz + __alpha
    Pdz = (Pdz.T / Pdz.sum(1)).T

    Pzv = n_zv + __beta
    Pzv = (Pzv.T / Pzv.sum(1)).T

    numpy.savetxt( "Pdz.txt", Pdz, fmt=str("%f") )
    numpy.savetxt( "Pzv.txt", Pzv, fmt=str("%f") )


# ldaメイン
def lda( data , K ):
    pylab.ion()
    # 尤度のリスト
    liks = []

    # 単語の種類数
    V = len(data[0])    # 語彙数
    D = len(data)       # 文書数

    # data内の単語を一列に並べる（計算しやすくするため）
    docs_dn = [ None for i in range(D) ]
    topics_dn = [ None for i in range(D) ]
    for d in range(D):
        docs_dn[d] = conv_to_word_list( data[d] )
        topics_dn[d] = numpy.random.randint( 0, K, len(docs_dn[d]) ) # 各単語にランダムでトピックを割り当てる

    # LDAのパラメータを計算
    n_dz, n_zv, n_z = calc_lda_param( docs_dn, topics_dn, K, V )


    for it in range(20):
        # メインの処理
        for d in range(D):
            N = len(docs_dn[d]) # 文書dに含まれる単語数
            for n in range(N):
                v = docs_dn[d][n]       # 単語のインデックス
                z = topics_dn[d][n]     # 単語に割り当てられているトピック


                # データを取り除きパラメータを更新
                n_dz[d][z] -= 1
                n_zv[z][v] -= 1
                n_z[z] -= 1

                # サンプリング
                z = sample_topic( d, v, n_dz, n_zv, n_z, K, V )

                # データをサンプリングされたクラスに追加してパラメータを更新
                topics_dn[d][n] = z
                n_dz[d][z] += 1
                n_zv[z][v] += 1
                n_z[z] += 1

        lik = calc_liklihood( data, n_dz, n_zv, n_z, K, V )
        liks.append( lik )
        print "対数尤度：", lik
        doc_dopics = numpy.argmax( n_dz , 1 )
        print "分類結果：", doc_dopics
        print "---------------------"


        # グラフ表示
        pylab.clf()
        pylab.subplot("121")
        pylab.title( "P(z|d)" )
        pylab.imshow( n_dz / numpy.tile(numpy.sum(n_dz,1).reshape(D,1),(1,K)) , interpolation="none" )
        pylab.subplot("122")
        pylab.title( "liklihood" )
        pylab.plot( range(len(liks)) , liks )
        pylab.draw()
        pylab.pause(0.1)

    save_model( n_dz, n_zv, n_z )
    pylab.ioff()
    pylab.show()

def main():
    data = numpy.loadtxt( "hist.txt" , dtype=numpy.int32)*10
    lda( data , 3 )

if __name__ == '__main__':
    main()

