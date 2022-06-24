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

import matplotlib
matplotlib.use("Agg")

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
import smtplib
import pandas as pd
import numpy as np


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
    elif country == 'uk':
        country_in = 'United Kingdom'

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

        for i in range(len(tester_list)):
            if tester_list['sum'][i] > 1:
                testers.append(tester_list['Disease'][i])

        get_result = np.zeros(shape=(len(testers), date_num))

        for m in range(date_num):
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

                for j in range(len(korea_filtered['var2'].tolist())):
                    cor_list.append(korea_filtered['var2'].tolist()[j])

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

        csv_date = []

        for i in range(0, date_num):
            csv_date.append(str(Date2 - timedelta(date_num) + timedelta(i)))

        if len(testers)>2:
            disease1 = testers[0]
            disease2 = testers[1]
            disease3 = testers[2]

            disease1_data = get_result[0]
            disease2_data = get_result[1]
            disease3_data = get_result[2]
        elif len(testers) == 2:
            disease1 = testers[0]
            disease2 = testers[1]
            disease3 = ''

            disease1_data = get_result[0]
            disease2_data = get_result[1]
            disease3_data = ''
        elif len(testers) == 1:
            disease1 = testers[0]
            disease2 = ''
            disease3 = ''

            disease1_data = get_result[0]
            disease2_data = ''
            disease3_data = ''
        elif len(testers) == 0:
            disease1 = ''
            disease2 = ''
            disease3 = ''

            disease1_data = ''
            disease2_data = ''
            disease3_data = ''

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
                                                                         'disease1_data': str(disease1_data.tolist()),
                                                                         'disease2_data': str(disease2_data.tolist()),
                                                                         'disease3_data': str(disease3_data.tolist()),
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

        graphs = []
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
                    text=sampled['variant2']
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
