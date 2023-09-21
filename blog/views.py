from django.shortcuts import render, redirect
from django.utils import timezone
from django.contrib import auth
from .models import Post
from django.contrib.auth.models import User
from django.http import HttpResponseRedirect
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime
from datetime import timedelta
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from wordcloud import WordCloud
from io import BytesIO
from plotly.offline import plot
from Bio import SeqIO
from bs4 import BeautifulSoup
from glob import glob
from django.http import HttpResponse, Http404

import requests
import os
import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
import smtplib
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET


def post_main(request):
    posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
    return render(request, 'blog/post_main.html', {'posts': posts})


def drw_country(request):
    country = request.GET.get('country')
    date = request.GET.get('date')
    country_in = ''

    if country == 'kor':
        country_in = 'Korea, Republic of'
    elif country == 'jap':
        country_in = 'Japan'
    elif country == 'chn':
        country_in = 'China'
    elif country == 'tha':
        country_in = 'Thailand'
    elif country == 'usa':
        country_in = 'United States'
    elif country == 'id':
        country_in = 'India'
    elif country == 'mx':
        country_in = 'Mexico'
    elif country == 'arg':
        country_in = 'Argentina'
    elif country == 'aus':
        country_in = 'Australia'

    if country_in != '':
        date_time_str = date
        today = datetime.strptime(date_time_str, '%Y-%m-%d')
        end_date = today - timedelta(0)
        start_date = end_date - timedelta(180)

        param_C = 1000
        param_gamma = 0.1

        date_num = 10

        Date1 = str(start_date)[:10]
        Date2 = str(end_date)[:10]

        Date1 = datetime.strptime(Date1, '%Y-%m-%d').date()
        Date2 = datetime.strptime(Date2, '%Y-%m-%d').date()

        tester_list = pd.read_csv(
            'blog/static/data/results/Predict_Result__' + str(
                end_date).replace('-', '')[:8] + '_Extra_without_corona.csv', index_col=0)

        tester_list["sum"] = tester_list.sum(axis=1)
        testers = []

        """
        for i in range(len(tester_list)):
            if tester_list['sum'][i] > 1:
                testers.append(tester_list['Disease'][i])


        get_result = np.zeros(shape=(len(testers), date_num))
        """

        tester_list = pd.read_csv(
                'blog/static/data/train_data/MEDISYS_All_Languages_WHOLE_' + str(
                    20211224) + '-' + str(20220622) + '.csv', index_col=0)

        testers = tester_list.T.index.tolist()

        get_result = np.zeros(shape=(len(testers), date_num))

        for m in range(1):
            start_date = str(Date1 - timedelta(m)).replace('-', '')[:8]
            end_date = str(Date2 - timedelta(m)).replace('-', '')[:8]

            train_data = pd.read_csv(
                'blog/static/data/train_data/MEDISYS_All_Languages_WHOLE_' + str(
                    start_date) + '-' + str(end_date) + '.csv', index_col=0)

            disease_list = train_data.columns.tolist()
            country_list = train_data.index.tolist()

            normalize_mode = 1
            cor_val = 0

            country_name = country_in

            if country_name in train_data.index.tolist():
                data1 = train_data
                data2 = data1.values.T

                if normalize_mode == 1:
                    min_max_scaler = preprocessing.MinMaxScaler()
                    data2 = min_max_scaler.fit_transform(data2)

                data = pd.DataFrame(data2, columns=country_list, index=disease_list).sort_index()

                corr = pd.DataFrame(data).corr(method='pearson')
                links = corr.stack().reset_index()
                links.columns = ['var1', 'var2', 'value']
                korea_filtered = links.loc[(links['value'] > cor_val) & (links['var1'] != links['var2']) \
                                           & (links['var1'] == country_name)]

                cor_list = []

                for n in range(len(korea_filtered['var2'].tolist())):
                    cor_list.append(korea_filtered['var2'].tolist()[n])

                cor_list.append(country_name)

                if len(korea_filtered) > 5:
                    data1 = train_data
                    data2 = pd.DataFrame(data1.T)
                    data2.columns = country_list
                    data3 = data2[data2.columns.intersection(cor_list)].T

                    for j in range(len(testers)):
                        label = []
                        disease_name = testers[j]

                        for k in range(len(data3.index.values)):
                            if data3[disease_name][k] == 0:
                                label.append(0)
                            else:
                                label.append(1)

                        label_data = pd.DataFrame(label, index=data3.index.values)

                        data4 = data3.T
                        data4.columns = range(data4.shape[1])

                        min_max_scaler = preprocessing.MinMaxScaler()
                        data4 = min_max_scaler.fit_transform(data4.values.tolist())
                        data4 = pd.DataFrame(data4, index=data3.T.index.values)
                        data4.columns = data3.index.values

                        data4 = data4.T

                        x_train = data4.drop([country_name])
                        x_train = x_train.T.drop([disease_name]).T
                        x_test = data4.T[country_name].T
                        x_test = x_test.drop([disease_name])

                        y_train = label_data.drop([country_name]).iloc[:, 0].tolist()

                        ####SVM
                        if sum(y_train) == 0:
                            asdf = 1
                        elif sum(y_train) == len(x_train):
                            asdf = 1
                        else:
                            classifier = OneVsRestClassifier(
                                SVC(kernel='rbf', C=param_C, gamma=param_gamma, probability=True)) \
                                .fit(x_train, y_train)
                            probability = classifier.predict_proba(pd.DataFrame(x_test).T)[:, 1][0]
                            get_result[j, date_num - 1 - m] = probability

        last_day_score = []
        for i in range(len(get_result)):
            last_day_score.append(get_result[i][-1])

        sorted_last_day_score = sorted(last_day_score, reverse=True)

        ranking_index = []
        ranking_index.append(last_day_score.index(sorted_last_day_score[0]))
        ranking_index.append(last_day_score.index(sorted_last_day_score[1]))
        ranking_index.append(last_day_score.index(sorted_last_day_score[2]))
        ranking_index.append(last_day_score.index(sorted_last_day_score[3]))
        ranking_index.append(last_day_score.index(sorted_last_day_score[4]))

        tester_pick = []
        tester_pick.append(testers[last_day_score.index(sorted_last_day_score[0])])
        tester_pick.append(testers[last_day_score.index(sorted_last_day_score[1])])
        tester_pick.append(testers[last_day_score.index(sorted_last_day_score[2])])
        tester_pick.append(testers[last_day_score.index(sorted_last_day_score[3])])
        tester_pick.append(testers[last_day_score.index(sorted_last_day_score[4])])

        testers = tester_pick

        get_result = np.zeros(shape=(len(testers), date_num))

        for m in range(10):
            start_date = str(Date1 - timedelta(m)).replace('-', '')[:8]
            end_date = str(Date2 - timedelta(m)).replace('-', '')[:8]

            train_data = pd.read_csv(
                'blog/static/data/train_data/MEDISYS_All_Languages_WHOLE_' + str(
                    start_date) + '-' + str(end_date) + '.csv', index_col=0)

            disease_list = train_data.columns.tolist()
            country_list = train_data.index.tolist()

            normalize_mode = 1
            cor_val = 0

            country_name = country_in

            if country_name in train_data.index.tolist():
                data1 = train_data
                data2 = data1.values.T

                if normalize_mode == 1:
                    min_max_scaler = preprocessing.MinMaxScaler()
                    data2 = min_max_scaler.fit_transform(data2)

                data = pd.DataFrame(data2, columns=country_list, index=disease_list).sort_index()

                corr = pd.DataFrame(data).corr(method='pearson')
                links = corr.stack().reset_index()
                links.columns = ['var1', 'var2', 'value']
                korea_filtered = links.loc[(links['value'] > cor_val) & (links['var1'] != links['var2']) \
                                           & (links['var1'] == country_name)]

                cor_list = []

                for n in range(len(korea_filtered['var2'].tolist())):
                    cor_list.append(korea_filtered['var2'].tolist()[n])

                cor_list.append(country_name)

                if len(korea_filtered) > 5:
                    data1 = train_data
                    data2 = pd.DataFrame(data1.T)
                    data2.columns = country_list
                    data3 = data2[data2.columns.intersection(cor_list)].T

                    for j in range(len(testers)):
                        label = []
                        disease_name = testers[j]
                        for k in range(len(data3.index.values)):
                            if data3[disease_name][k] == 0:
                                label.append(0)
                            else:
                                label.append(1)

                        label_data = pd.DataFrame(label, index=data3.index.values)

                        data4 = data3.T
                        data4.columns = range(data4.shape[1])

                        min_max_scaler = preprocessing.MinMaxScaler()
                        data4 = min_max_scaler.fit_transform(data4.values.tolist())
                        data4 = pd.DataFrame(data4, index=data3.T.index.values)
                        data4.columns = data3.index.values

                        data4 = data4.T

                        x_train = data4.drop([country_name])
                        x_train = x_train.T.drop([disease_name]).T
                        x_test = data4.T[country_name].T
                        x_test = x_test.drop([disease_name])

                        y_train = label_data.drop([country_name]).iloc[:, 0].tolist()

                        ####SVM
                        if sum(y_train) == 0:
                            asdf = 1
                        elif sum(y_train) == len(x_train):
                            asdf = 1
                        else:
                            classifier = OneVsRestClassifier(
                                SVC(kernel='rbf', C=param_C, gamma=param_gamma, probability=True)) \
                                .fit(x_train, y_train)
                            probability = classifier.predict_proba(pd.DataFrame(x_test).T)[:, 1][0]
                            get_result[j, date_num - 1 - m] = probability

        last_day_score = []
        for i in range(len(get_result)):
            last_day_score.append(get_result[i][-1])

        sorted_last_day_score = sorted(last_day_score, reverse=True)

        ranking_index = []
        ranking_index.append(last_day_score.index(sorted_last_day_score[0]))
        ranking_index.append(last_day_score.index(sorted_last_day_score[1]))
        ranking_index.append(last_day_score.index(sorted_last_day_score[2]))
        ranking_index.append(last_day_score.index(sorted_last_day_score[3]))
        ranking_index.append(last_day_score.index(sorted_last_day_score[4]))

        csv_date = []

        for i in range(0, date_num):
            csv_date.append(str(Date2 - timedelta(date_num) + timedelta(i)))

        disease1 = testers[ranking_index[0]]
        disease2 = testers[ranking_index[1]]
        disease3 = testers[ranking_index[2]]
        disease4 = testers[ranking_index[3]]
        disease5 = testers[ranking_index[4]]

        disease1_data = get_result[ranking_index[0]]
        disease2_data = get_result[ranking_index[1]]
        disease3_data = get_result[ranking_index[2]]
        disease4_data = get_result[ranking_index[3]]
        disease5_data = get_result[ranking_index[4]]

        train_data = pd.read_csv(
            'blog/static/data/train_data/MEDISYS_All_Languages_WHOLE_' + str(
                start_date) + '-' + str(end_date) + '.csv', index_col=0)

        data1 = train_data
        data2 = data1.values.T
        min_max_scaler = preprocessing.MinMaxScaler()
        data2 = min_max_scaler.fit_transform(data2)

        data = pd.DataFrame(data2, columns=data1.index.tolist(), index=list(data1)).sort_index()

        corr = pd.DataFrame(data).corr(method='pearson')
        word_freq = corr[country_in].to_dict()
        wordcloud = WordCloud().generate_from_frequencies(word_freq)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.imshow(wordcloud)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
        buf.close()

        return render(request, 'blog/disease_risk_weight_country.html', {'xaxis': str(csv_date),
                                                                         'set_date': date,
                                                                         'disease1': disease1,
                                                                         'disease2': disease2,
                                                                         'disease3': disease3,
                                                                         'disease4': disease4,
                                                                         'disease5': disease5,
                                                                         'disease1_data': str(disease1_data.tolist()),
                                                                         'disease2_data': str(disease2_data.tolist()),
                                                                         'disease3_data': str(disease3_data.tolist()),
                                                                         'disease4_data': str(disease4_data.tolist()),
                                                                         'disease5_data': str(disease5_data.tolist()),
                                                                         'image_base64': image_base64,
                                                                         'country': country_in,})

    else:
        posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
        return render(request, 'blog/disease_risk_weight_country.html', {'posts': posts})


def drw_disease(request):
    disease = request.GET.get('disease')
    date = request.GET.get('date')

    if disease != None:
        date_time_str = date
        today = datetime.strptime(date_time_str, '%Y-%m-%d')
        end_date = today - timedelta(0)
        start_date = end_date - timedelta(180)

        Date1 = str(start_date)[:10]
        Date2 = str(end_date)[:10]

        Date1 = datetime.strptime(Date1, '%Y-%m-%d').date()
        Date2 = datetime.strptime(Date2, '%Y-%m-%d').date()

        start_date = str(Date1).replace('-', '')[:8]
        end_date = str(Date2).replace('-', '')[:8]

        train_data = pd.read_csv(
            'blog/static/data/train_data/MEDISYS_All_Languages_WHOLE_' + str(
                start_date) + '-' + str(end_date) + '.csv', index_col=0)

        data = [
                dict(type='choropleth',
                     locations=train_data[disease].index.tolist(),
                     locationmode='country names',
                     z=train_data[disease].values.tolist(),
                     # text = data1.iloc[:,2],
                     colorbar=dict(title='# of articles', lenmode='pixels', len=175, yanchor='bottom', y=0))]

        layout = dict(title='# of ' + disease + ' related articles from ' + start_date + ' ~ ' + end_date,
                      geo=dict(showframe=False,
                               projection={'type': 'equirectangular'},),
                      height=500,
                      width=700,)
        map2 = dict(data=data, layout=layout)
        plot_div = plot(map2, output_type='div')

        data1 = train_data
        data2 = data1.values
        min_max_scaler = preprocessing.MinMaxScaler()
        data2 = min_max_scaler.fit_transform(data2).T

        data = pd.DataFrame(data2, columns=data1.index.tolist(), index=list(data1)).sort_index()

        corr = pd.DataFrame(data.T).corr(method='pearson')

        word_freq = corr[disease].to_dict()
        wordcloud = WordCloud().generate_from_frequencies(word_freq)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.imshow(wordcloud)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
        buf.close()

        return render(request, 'blog/disease_risk_weight_disease.html', context={'plot_div': plot_div,
                                                                                 'disease': disease,
                                                                                 'set_date': date,
                                                                                 'image_base64': image_base64,})

    else:
        posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
        return render(request, 'blog/disease_risk_weight_disease.html', {'posts': posts})


def drw_info(request):
    posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
    return render(request, 'blog/disease_risk_weight_info.html', {'posts': posts})


def seq_analyze(request):
    posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
    return render(request, 'blog/seq_analyze.html', {'posts': posts})


def seq_analyze_tables(request):
    legend = request.GET.get('legend')
    samples = request.GET.get('samples')

    if legend != None:
        data = pd.read_csv('blog/static/data/meta_data_ForPlatForm.csv', index_col=0)

        label_info = pd.DataFrame({'Year': data['year'], 'labels': data['label']})
        label_info['count'] = 1
        result = label_info.pivot_table(index=['labels'], columns=['Year'], values='count', fill_value=0,
                                        aggfunc=np.sum)
        result['count'] = result.sum(axis=1)
        result = result.loc[result['count'] > 1120]

        sampled = data.loc[data['label'].isin(result.index.tolist())]
        sampled = sampled.sample(n=int(samples), random_state=1)

        legend_list = np.asarray(sampled[legend])

        unique_labels = set(legend_list)
        colors = []

        for i in range(len(legend_list)):
            colors.append(list(unique_labels).index(legend_list[i]))

        names = {k: str(v) for k, v in zip(set(colors), set(legend_list))}

        df = pd.DataFrame({'x': sampled['x1'],
                           'y': sampled['x2'],
                           'color': colors})

        fig = go.Figure()

        for c in df['color'].unique():
            df_color = df[df['color'] == c]
            fig.add_trace(
                go.Scatter(
                    x=df_color['x'],
                    y=df_color['y'],
                    name=names[c],
                    mode='markers',
                    showlegend=True,
                    text=sampled['info']
                )
            )

        fig.update_traces(marker=dict(size=9,
                                      line=dict(width=2,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01),)

        if 'sample1' in request.GET:
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[50],
                    name='Sample1',
                    mode='markers',
                    showlegend=True,
                    text='2023.01.15/Seoul',
                    marker=dict(color='greenyellow',
                                size=20,
                                line=dict(width=2,
                                          color='DarkSlateGrey'))
                )
            )

        if 'sample2' in request.GET:
            fig.add_trace(
                go.Scatter(
                    x=[60],
                    y=[20],
                    name='Sample2',
                    mode='markers',
                    showlegend=True,
                    text='2023.03.10/Daejeon',
                    marker=dict(color='greenyellow',
                                size=20,
                                line=dict(width=2,
                                          color='DarkSlateGrey'))
                )
            )

        if 'sample3' in request.GET:
            fig.add_trace(
                go.Scatter(
                    x=[30],
                    y=[0],
                    name='Sample3',
                    mode='markers',
                    showlegend=True,
                    text='2023.04.21/Busan',
                    marker=dict(color='greenyellow',
                                size=20,
                                line=dict(width=2,
                                          color='DarkSlateGrey'))
                )
            )

            fig.update_traces(selector=dict(mode='markers'))

        plot_div = plot(fig, output_type='div')

        label_info = pd.DataFrame(
            {'period': sampled['period'], 'label': sampled['label'], 'variant2': sampled['variant2'],
             'who_region': sampled['who_region']})
        label_info['count'] = 1
        result = label_info.pivot_table(index=[legend], columns=['period'], values='count', fill_value=0,
                                        aggfunc=np.sum)
        result['count'] = result.sum(axis=1)

        return render(request, 'blog/seq_analyze_tables.html', context={'plot_div': plot_div,
                                                                         'legend': legend,
                                                                         'samplesa': samples,
                                                                         'table': result.to_html(classes="table .table-striped")})

    else:
        posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
        return render(request, 'blog/seq_analyze_tables.html', {'posts': posts})


"""
def seq_analyze_tables(request):
    legend = request.GET.get('legend')
    samples = request.GET.get('samples')
    sequence = request.GET.get('sequence')
    notReference_data_spike = np.zeros(61)

    if sequence != None:
        reference_seq = "blog/static/data/sequence/MT019529.1_reference_sequence_.fasta"
        target_seq = sequence
        output = ""

        data = str(SeqIO.read(reference_seq, "fasta").seq).upper()
        output = output + ">" + SeqIO.read(reference_seq, "fasta").id + "\n" + data + "\n"
        output = output + ">Input\n" + target_seq + "\n"
        outfile = open("blog/static/data/sequence/input.txt", "w+")
        outfile.write(output)
        outfile.close()

        f = "blog/static/data/sequence/input.txt"

        os.system(
            '/Users/juhyeon/MAFFT/mafft-mac/mafft.bat --auto ' + f + ' > blog/static/data/sequence/input_align.txt')

        f = "blog/static/data/sequence/input_align.txt"
        output = ""
        data = str(list(SeqIO.parse(f, 'fasta'))[-1].seq)[21562:25384].upper()
        output = output + ">Input\n" + data + "\n"
        outfile = open("blog/static/data/sequence/input_spike.txt", "w+")
        outfile.write(output)
        outfile.close()

        f = "blog/static/data/sequence/input_spike.txt"

        pre_file = ""
        test_seq = str(SeqIO.read(f, 'fasta').seq)
        pre_seq = test_seq.replace('-', '0') \
            .replace('N', '0').replace('Y', '0').replace('M', '0').replace('S', '0').replace('K', '0') \
            .replace('R', '0').replace('W', '0').replace('D', '0').replace('V', '0').replace('H', '0') \
            .replace('B', '0')

        outfile = open("blog/static/data/sequence/input_preprocessed.txt", 'w+')
        outfile.write(pre_seq)
        outfile.close()

        codonTable = ["UUU", "UUC", "UUA", "UUG", "CUU", "CUC", "CUA", "CUG", "AUU", "AUC", "AUA", "AUG",
                      "GUU", "GUC", "GUA", "GUG", "UCU", "UCC", "UCA", "UCG", "CCU", "CCC", "CCA", "CCG",
                      "ACU", "ACC", "ACA", "ACG", "GCU", "GCC", "GCA", "GCG", "UAU", "UAC",
                      "CAU", "CAC", "CAA", "CAG", "AAU", "AAC", "AAA", "AAG", "GAU", "GAC", "GAA", "GAG",
                      "UGU", "UGC", "UGG", "CGU", "CGC", "CGA", "CGG", "AGU", "AGC", "AGA", "AGG",
                      "GGU", "GGC", "GGA", "GGG"]

        f = "blog/static/data/sequence/input_preprocessed.txt"

        txt_data = open(f, 'r').read()

        for i in range(int(len(txt_data) / 3)):
            if txt_data.replace('T', 'U')[0 + 3 * i:3 + 3 * i] not in ['UAA', 'UGA',
                                                                       'UAG'] and '0' not in txt_data.replace('T', 'U')[
                                                                                             0 + 3 * i:3 + 3 * i]:
                notReference_data_spike[codonTable.index(txt_data.replace('T', 'U')[0 + 3 * i:3 + 3 * i])] += 1

    if legend != None:
        if notReference_data_spike[0] != 0:
            max_prm = [63.0, 40.0, 55.0, 73.0, 47.0, 20.0, 50.0, 47.0, 54.0, 25.0, 26.0, 58.0, 51.0, 23.0, 21.0, 55.0,
                       40.0, 17.0, 37.0, 5.0, 30.0, 10.0, 27.0, 4.0, 47.0, 29.0, 43.0, 6.0, 44.0, 11.0, 30.0, 4.0, 45.0,
                       33.0, 25.0, 37.0, 49.0, 40.0, 59.0, 37.0, 44.0, 30.0, 45.0, 21.0, 36.0, 18.0, 61.0, 55.0, 51.0,
                       11.0, 7.0, 3.0, 4.0, 31.0, 16.0, 41.0, 24.0, 50.0, 17.0, 20.0, 9.0]
            min_prm = [28.0, 10.0, 14.0, 12.0, 15.0, 5.0, 2.0, 1.0, 23.0, 8.0, 9.0, 5.0, 13.0, 4.0, 4.0, 4.0, 11.0, 4.0,
                       15.0, 0.0, 5.0, 0.0, 9.0, 0.0, 15.0, 4.0, 23.0, 1.0, 3.0, 3.0, 7.0, 0.0, 5.0, 3.0, 6.0, 0.0,
                       21.0, 7.0, 22.0, 14.0, 21.0, 6.0, 4.0, 3.0, 12.0, 3.0, 13.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 10.0,
                       2.0, 9.0, 4.0, 6.0, 4.0, 3.0, 1.0]

            scaled_seq = []

            for i in range(len(notReference_data_spike)):
                scaled_seq.append((notReference_data_spike[i] - min_prm[i]) / (max_prm[i] - min_prm[i]))

            new_model = tf.keras.models.load_model('blog/static/data/sequence/covid.h5')
            predicted_vector = new_model.predict(np.asarray(scaled_seq).reshape(1,61))

        #data = pd.read_csv('blog/static/data/meta_data_ForPlatForm.csv', index_col=0)
        data = pd.read_csv('blog/static/data/meta_data_bmgf.csv', index_col=0, encoding='cp949')

        label_info = pd.DataFrame({'Year': data['year'], 'labels': data['label']})
        label_info['count'] = 1
        result = label_info.pivot_table(index=['labels'], columns=['Year'], values='count', fill_value=0,
                                        aggfunc=np.sum)
        result['count'] = result.sum(axis=1)
        result = result.loc[result['count'] > 1120]

        graphs = []
        #sampled = data.loc[data['label'].isin(result.index.tolist())]
        #sampled = sampled.sample(n=int(samples), random_state=1)
        sampled = data.sample(n=int(samples), random_state=1)

        legend_list = np.asarray(sampled[legend])

        unique_labels = set(legend_list)
        colors = []

        for i in range(len(legend_list)):
            colors.append(list(unique_labels).index(legend_list[i]))

        names = {k: str(v) for k, v in zip(set(colors), set(legend_list))}

        df = pd.DataFrame({'x': sampled['x1'],
                           'y': sampled['x2'],
                           'color': colors})

        fig = go.Figure()

        for c in df['color'].unique():
            df_color = df[df['color'] == c]
            fig.add_trace(
                go.Scatter(
                    x=df_color['x'],
                    y=df_color['y'],
                    name=names[c],
                    mode='markers',
                    showlegend=True,
                    #text=sampled['variant']
                )
            )

        fig.update_traces(marker=dict(size=9,
                                      line=dict(width=2,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01),)

        if notReference_data_spike[0] != 0:
            fig.add_trace(
                go.Scatter(
                    x=pd.Series(predicted_vector[0][0]),
                    y=pd.Series(predicted_vector[0][1]),
                    name='Input Sequence',
                    mode='markers',
                    showlegend=True,
                    #text='Input Sequence',
                    marker=dict(color='greenyellow',
                                size=20,
                                line=dict(width=2,
                                          color='DarkSlateGrey'))
                )
            )

        fig.update_traces(selector=dict(mode='markers'))

        plot_div = plot(fig, output_type='div')
        label_info = pd.DataFrame(
            {'period': sampled['period'], 'label': sampled['label'], 'variant': sampled['variant']})
        label_info['count'] = 1
        result = label_info.pivot_table(index=['variant'], columns=['period'], values='count', fill_value=0,
                                        aggfunc=np.sum)
        result['count'] = result.sum(axis=1)

        return render(request, 'blog/seq_analyze_tables.html', context={'plot_div': plot_div,
                                                                         'legend': legend,
                                                                         'samplesa': samples,
                                                                         'table': result.to_html(classes="table .table-striped")})

    else:
        posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
        return render(request, 'blog/seq_analyze_tables.html', {'posts': posts})
"""


def logout(request):
    response = redirect('/')
    auth.logout(request)
    return response


def login(request):
    after_url = request.GET.get('next', '/')

    if request.method == "POST":
        username = request.POST['username'].strip()
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)

        if user is not None:
            auth.login(request, user)
            response = HttpResponseRedirect(after_url)
            return response
        else:
            return render(request, 'blog/post_main.html', {'error': '아이디와 비밀번호를 확인해주세요.'})

    else:
        return render(request, '/')


@csrf_exempt
def signup(request):
    if request.method == "POST":
        if request.POST["password1"] == request.POST["password2"]:

            if request.POST["usertype"] == "consumer":
                user = User.objects.create_user(username=request.POST["username"].strip(),
                                                password=request.POST["password1"].strip(),
                                                email=request.POST["username"].strip(),
                                                realname=request.POST["realname"].strip(),
                                                workplace=request.POST["workplace"],)
            elif request.POST["usertype"] == "expert":
                user = User.objects.create_user(username=request.POST["username"].strip(),
                                                password=request.POST["password1"].strip(),
                                                email=request.POST["username"].strip(),
                                                realname=request.POST["realname"].strip(),
                                                workplace=request.POST["workplace"],)

            auth.login(request, user)
            return redirect('/')

        elif request.POST["password1"] != request.POST["password2"]:
            return render(request, 'blog/signup.html', {'error': '비밀번호가 일치하지 않습니다. 확인해주세요.'})
    else:
        return render(request, 'blog/signup.html')


@csrf_exempt
def pswdmod(request):
    if request.method == "POST":
        username = request.POST["username"]
        key = 'heyvcc'

        try:
            User.objects.get(username=username)
            enc = []

            for i in range(len(username)):
                key_c = key[i % len(key)]
                enc_c = chr((ord(username[i]) + ord(key_c)) % 256)
                enc.append(enc_c)

            verify_code = base64.urlsafe_b64encode("".join(enc).encode()).decode().replace('=', '')

            if request.POST["verification"].strip() != verify_code.strip():

                gmail_sender = 'vccnet.owners@gmail.com'
                gmail_passwd = 'xhtmuimafstrqdsi'
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.ehlo()
                server.starttls()
                server.login(gmail_sender, gmail_passwd)

                msg = MIMEMultipart('alternative')
                msg['Subject'] = '[비밀번호 변경] 인증코드를 복사하여 입력해주세요.'
                msg['From'] = gmail_sender
                msg['To'] = username

                html = """\
                <!DOCTYPE html>
                <html>
                <body>

                <p>아래 인증 코드를 복사해서 입력해주세요.</p>
                인증코드<input type="text" style="width:450px;" value=""" + verify_code.strip() + """>
                <br>
                <br>
                <br>
                <br>
                <p>드래그 되지 않을 시</p><br>
                인증코드: """ + verify_code.strip() + """
                <p></p>

                </body>
                </html>
                """

                part2 = MIMEText(html, 'html')

                msg.attach(part2)

                server.sendmail(gmail_sender, [username], msg.as_string())
                server.quit()

                if request.POST["verification"].strip() != '':
                    errorMsg = '인증코드가 일치하지 않습니다. 인증 코드를 확인해주세요.'

                else:
                    errorMsg = username + '로 인증코드가 발송 되었습니다. 인증 코드를 입력해주세요.'

                return render(request, 'blog/post_main.html', {'username': username,
                                                             'pswdmoderror': errorMsg, })

            if request.POST["password1"] == request.POST["password2"]:
                u = User.objects.get(username__exact=request.POST["username"].strip())
                u.set_password(request.POST["password1"].strip())
                u.save()

                auth.login(request, u)
                return redirect('/')

            elif request.POST["password1"] != request.POST["password2"]:
                return render(request, 'blog/post_main.html', {'pswdmoderror': '비밀번호가 일치하지 않습니다. 확인해주세요.'})
        except:
            return render(request, 'blog/post_main.html', {'pswdmoderror': '존재하지 않는 계정입니다. 확인해주세요.'})
    else:
        return render(request, 'blog/post_main.html')


def bert(request):
    return render(request,'blog/bert.html')


@csrf_exempt
def report(request):
    df_report = pd.read_csv('df_WHO_result_for_Django.csv', encoding="ISO-8859-1", index_col=0)
    df_ref = pd.read_csv('WHO_url_list.csv', encoding="ISO-8859-1")
    df = pd.merge(df_report, df_ref, left_on='ref', right_on='name', how="left").drop_duplicates()

    if request.method == 'POST':
        selected_disease = request.POST['disease']
        selected_country = request.POST['country']
        selected_year = request.POST['year']
        selected_report = df[(df['country']==selected_country)&(df['disease']==selected_disease)&(df['year']==int(selected_year))]
        result_cases = list(selected_report.sort_values('month')['case'])
        result_colors = [i.replace('up', '#ff0000').replace('down', '#0000ff') for i in list(selected_report.sort_values('month')['state'])]
        result_contents = list(df['content'])
        result_refs = list(df['ref'])
        result_urls = list(df['url'])

        selected_return = {'selected_disease': selected_disease, 'selected_country': selected_country, 'selected_year': selected_year, 'result_cases':result_cases,
                           'result_colors': result_colors, 'result_contents': result_contents, 'result_refs': result_refs, 'result_urls': result_urls
                           }

        return render(request, 'blog/bert_graph.html', selected_return)
    else:
        return render(request, 'blog/bert_report.html')


def about(request):
    return render(request,'blog/bert_info.html')


def dengue(request):
    return render(request,'blog/dengue.html')


def dengue_report(request):
    return render(request,'blog/dengue_report.html')


def medisys_about(request):
    return render(request,'blog/medisys_about.html')


def medisys_crawl(request):
    disease = request.GET.get('disease')

    if disease != None:
        xml_list = glob("blog/static/data/crawl/*")
        disease_list = [disease]

        date = []
        links = []
        georss = []
        check_disease = []

        for disease in disease_list:
            xml_file = "blog/static/data/crawl/" + disease + ".xml"

            tree = ET.parse(xml_file, ET.XMLParser(encoding='utf-8'))
            root = tree.getroot()

            for k in range(len(root)):
                rss_checker = 0

                for x in range(len(root[k])):
                    if root[k][x].tag == 'pubDate':
                        pubDate_pointer = x

                    if root[k][x].tag == 'link':
                        link_pointer = x

                    if root[k][x].tag == 'georss':
                        georss_pointer = x
                        rss_checker = 1

                if rss_checker == 1:
                    check_disease.append(disease)
                    date.append(root[k][pubDate_pointer].text[12:16] + root[k][pubDate_pointer].text[8:11].replace('Jan',
                                                                                                                   '01').replace(
                        'Feb', '02').replace('Mar', '03').replace('Apr', '04').replace('May', '05').replace('Jun',
                                                                                                            '06').replace(
                        'Jul', '07').replace('Aug', '08').replace('Sep', '09').replace('Oct', '10').replace('Nov',
                                                                                                            '11').replace(
                        'Dec', '12') + root[k][pubDate_pointer].text[5:7])
                    links.append(root[k][link_pointer].text)
                    georss.append(root[k][georss_pointer].text)

        df = pd.DataFrame({'Disease': check_disease, 'Date': date, 'Link': links, 'Georss': georss})

        return render(request, 'blog/medisys_crawl.html', context={
                                                                   'table': df.to_html(classes="table .table-striped")})

    return render(request,'blog/medisys_crawl.html')


def medisys_crawl_func(request):
    url = 'http://medisys.newsbrief.eu/medisys/homeedition/en/home.html'
    source_code = requests.get(url, verify=False)
    plain_text = source_code.text
    soup = BeautifulSoup(plain_text, 'lxml')

    items = []

    content1 = soup.find('li', {'id': 'AllDiseasesA-D'})
    content2 = soup.find('li', {'id': 'AllDiseasesE-I'})
    content3 = soup.find('li', {'id': 'AllDiseasesJ-Q'})
    content4 = soup.find('li', {'id': 'AllDiseasesR-Z'})

    for link in content1.select('ul > li'):
        items.append(link.string)
    for link in content2.select('ul > li'):
        items.append(link.string)
    for link in content3.select('ul > li'):
        items.append(link.string)
    for link in content4.select('ul > li'):
        items.append(link.string)

    for i in range(1, len(items)):
        if items[i] != 'Encephalitis':
            xml = 'http://medisys.newsbrief.eu/rss?type=category&id=' + items[i].replace(' ', '').replace('ZikaVirus',
                                                                                                          'Zika') + '&language=all'
            src = requests.get(xml)
            src_re = src.text.replace('georss:point', 'georss')

            xml_file = open(
                'blog/static/data/crawl/' + items[
                    i] + '.xml', 'w')
            xml_file.write(src_re)
            xml_file.close()

    xml_list = glob("blog/static/data/crawl/*")
    disease_list = []

    for xml in xml_list:
        disease_list.append(xml[23:-4])

    if 'Healthcare Associated Infections' in disease_list:
        disease_list.remove('Healthcare Associated Infections')

    if 'Yersinia Pestis (Plague)' in disease_list:
            disease_list.remove('Yersinia Pestis (Plague)')

    def find_str(s, char):
        index = 0

        if char in s:
            c = char[0]
            for ch in s:
                if ch == c:
                    if s[index:index+len(char)] == char:
                        return index

                index += 1

        return -1

    for disease in disease_list:
        xml_file = "blog/static/data/crawl/" + disease + ".xml"
        try:
            tree = ET.ElementTree(file=xml_file)
            root = tree.getroot()

            xmlstr = ET.tostring(root, encoding='utf8', method='xml').decode('utf-8')

            rm_str = str(xmlstr[:find_str(xmlstr, '<rss')]) + str('<data>') + str(
                xmlstr[find_str(xmlstr, '<item>'):]) + str('</data>')
            rm_str = rm_str.replace('</channel>', '').replace('</rss>', '').replace('ns0:', 'ns0').replace('ns1:',
                                                                                                           'ns1').replace(
                '\n', '')

            text_file = open(xml_file, 'w')
            text_file.write(rm_str)
            text_file.close()

        except:
            print('Error occured while opening ' + xml_file)

    date = []
    links = []
    georss = []
    check_disease = []

    for disease in disease_list:
        xml_file = "blog/static/data/crawl/" + disease + ".xml"

        tree = ET.parse(xml_file, ET.XMLParser(encoding='utf-8'))
        root = tree.getroot()

        for k in range(len(root)):
            rss_checker = 0

            for x in range(len(root[k])):
                if root[k][x].tag == 'pubDate':
                    pubDate_pointer = x

                if root[k][x].tag == 'link':
                    link_pointer = x

                if root[k][x].tag == 'georss':
                    georss_pointer = x
                    rss_checker = 1

            if rss_checker == 1:
                check_disease.append(disease)
                date.append(root[k][pubDate_pointer].text[12:16] + root[k][pubDate_pointer].text[8:11].replace('Jan',
                                                                                                               '01').replace(
                    'Feb', '02').replace('Mar', '03').replace('Apr', '04').replace('May', '05').replace('Jun',
                                                                                                        '06').replace(
                    'Jul', '07').replace('Aug', '08').replace('Sep', '09').replace('Oct', '10').replace('Nov',
                                                                                                        '11').replace(
                    'Dec', '12') + root[k][pubDate_pointer].text[5:7])
                links.append(root[k][link_pointer].text)
                georss.append(root[k][georss_pointer].text)

    df = pd.DataFrame({'0. Disease': check_disease, '1. FullDate': date, '2. Link': links, '3. georss': georss})
    df.to_csv('blog/static/data/medisys_csv.csv')

    file_path = 'blog/static/data/medisys_csv.csv'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
    raise Http404


# (sy)
def vaers_about(request):
    return render(request,'blog/vaers_about.html')

@csrf_exempt
def vaers_analyze(request):
    df = pd.read_csv('blog/static/data/df_dbscan_covid_symptom.csv', encoding="ISO-8859-1")

    if request.method == 'POST':
        sex = request.POST['sex']
        age = request.POST['age']
        vax_mf = request.POST['vax_manu']
        feature = request.POST['feature']

        df_user = df[(df['SEX'] == sex) & (df['New_age'] == age) & (df['VAX_MANU'] == vax_mf)]
        top_sym_list = list(df_user['symptom'].value_counts()[:10].index)

        df_sp = df[(df['SEX'] == sex) & (df['New_age'] == age)]

        if feature == 'VAX_MANU':
            df_sp = df[(df['SEX'] == sex) & (df['New_age'] == age)]
            color_list = ['#379df0', '#1ac422','#f531b7']
            feature_list = ['JANSSEN', 'PFIZER\BIONTECH', 'MODERNA']
            title = 'The 1-10 most common symptoms of vaccine adverse events by vaccine manufacturer'

        if feature == 'SEX':
            df_sp = df[(df['VAX_MANU'] == vax_mf) & (df['New_age'] == age)]
            color_list = ['#f531b7', '#379df0']
            feature_list = ['F', 'M']
            title = 'The 1-10 most common symptoms of vaccine adverse events by gender'

        labels = top_sym_list
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        plt.rc('xtick', labelsize=15)
        fig, ax = plt.subplots(figsize=(15, 15), subplot_kw=dict(polar=True))

        def add_to_radar(df_sp, labels, feature_list, color_list):
            for i in range(len(feature_list)):
                color = color_list[i]
                df_sp_f = df_sp[df_sp[feature] == list(set(df_sp[feature]))[i]]
                print(list(set(df_sp[feature]))[i], len(df_sp_f))
                count_list = []
                for sym in labels:
                    count = len(df_sp_f[(df_sp_f['symptom'] == sym)])
                    count_list.append(count)
                print(count_list)
                count_list += count_list[:1]
                ax.plot(angles, count_list, color=color, linewidth=1, label=list(set(df_sp[feature]))[i])
                ax.fill(angles, count_list, color=color, alpha=0.25)

        add_to_radar(df_sp, labels, feature_list, color_list)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_thetagrids(np.degrees(angles[:-1]), labels)

        for label, angle in zip(ax.get_xticklabels(), angles):
            if angle in (0, np.pi):
                label.set_horizontalalignment('center')
            elif 0 < angle < np.pi:
                label.set_horizontalalignment('left')
            else:
                label.set_horizontalalignment('right')

        ax.set_rlabel_position(180 / num_vars)
        ax.set_title(title, y=1.08, size=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0), fontsize=14)

        fig.savefig('blog/static/favicon/vaers1.png', dpi=500)

        df_user = df[(df['SEX'] == sex) & (df['New_age'] == age) & (df['VAX_MANU'] == vax_mf)]
        top_sym_list = list(df_user['symptom'].value_counts()[10:20].index)

        if feature == 'VAX_MANU':
            df_sp = df[(df['SEX'] == sex) & (df['New_age'] == age)]
            color_list = ['#f531b7', '#1ac422', '#379df0']
            feature_list = [vax_mf]
            feature_list = feature_list + list(set(df[feature]) - set([vax_mf]))
            title = 'The 11-20 most common symptoms of vaccine adverse events by vaccine manufacturer'

        if feature == 'SEX':
            df_sp = df[(df['VAX_MANU'] == vax_mf) & (df['New_age'] == age)]
            color_list = ['#f531b7', '#379df0']
            feature_list = ['F', 'M']
            title = 'The 11-20 most common symptoms of vaccine adverse events by gender'

        labels = top_sym_list
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(13, 13), subplot_kw=dict(polar=True))

        def add_to_radar(df_sp, labels, feature_list, color_list):
            for i in range(len(feature_list)):
                color = color_list[i]
                df_sp_f = df_sp[df_sp[feature] == list(set(df_sp[feature]))[i]]
                print(list(set(df_sp[feature]))[i], len(df_sp_f))
                count_list = []
                for sym in labels:
                    count = len(df_sp_f[(df_sp_f['symptom'] == sym)])
                    count_list.append(count)
                print(count_list)
                count_list += count_list[:1]
                ax.plot(angles, count_list, color=color, linewidth=1, label=list(set(df_sp[feature]))[i])
                ax.fill(angles, count_list, color=color, alpha=0.25)

        add_to_radar(df_sp, labels, feature_list, color_list)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_thetagrids(np.degrees(angles[:-1]), labels)

        ax.set_rlabel_position(180 / num_vars)
        ax.set_title(title, y=1.08, size=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0), fontsize=15)

        fig.savefig('blog/static/favicon/vaers2.png', dpi=500)

        return render(request, 'blog/vaers_analyze_result.html')
    else:
        return render(request, 'blog/vaers_analyze.html')
