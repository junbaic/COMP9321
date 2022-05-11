from flask import Flask, request, jsonify, make_response, Response
from flask_restx import Resource, Api, reqparse
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import requests
import re
import json
from sqlalchemy import func
import pandas as pd
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.config['SQLALCHEMY_ECHO'] = True
app.config['JSON_SORT_KEYS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///YOURZID.db'
db = SQLAlchemy(app)
api = Api(app, version='1.0', title='API Doc', description='', doc='/')


class Actor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    lastUpdate = db.Column(db.String(255), nullable=False)
    # lastUpdate = db.Column(db.DateTime, default=datetime.utcnow)
    name = db.Column(db.String(255), nullable=False)
    gender = db.Column(db.String(255), nullable=True)
    country = db.Column(db.String(255), nullable=True)
    birthday = db.Column(db.String(255), nullable=True)
    deathday = db.Column(db.String(255), nullable=True)
    shows = db.Column(db.String(255), nullable=True)

    def __repr__(self):
        print("id:{0} name:{1}".format(self.id, self.name))


db.create_all()


def dbInit():
    def addInitActor(name: str):
        name = "".join([c if c.isalnum() else " " for c in name])  # 用空格替换所有非英语字符
        response = requests.get(url='https://api.tvmaze.com/search/people?q={0}'.format(name),
                                timeout=60)

        actorList = json.loads(response.text)
        for actor in actorList:
            # 忽略大小写、以及除字母、空格、数字之外的任何字符
            actorName = str.lower(actor['person']['name'])
            actorName = str(re.sub(r'[^\w\s\d]+', ' ', actorName)).strip()
            if (str.lower(name) == actorName):
                id = actor['person']['id']
                name = actor['person']['name']
                gender = actor['person']['gender']
                if (actor['person']['country'] is not None):
                    country = actor['person']['country']['name']
                else:
                    country = None
                birthday = actor['person']['birthday']
                if (birthday is not None):
                    birthdayItemList = str(birthday).split('-')
                    birthday = '{0}-{1}-{2}'.format(birthdayItemList[2], birthdayItemList[1], birthdayItemList[0])
                deathday = actor['person']['deathday']
                if (deathday is not None):
                    deathdayItemList = str(deathday).split('-')
                    deathday = '{0}-{1}-{2}'.format(deathdayItemList[2], deathdayItemList[1], deathdayItemList[0])
                response = requests.get(url='https://api.tvmaze.com/people/{0}/castcredits?embed=show'.format(id),
                                        timeout=60)
                shows = ','.join(
                    ['show {0}'.format(item['_embedded']['show']['id']) for item in json.loads(response.text)])
                lastUpdate = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                # 无则新增，有则更新
                if (Actor.query.filter_by(id=id).first() == None):
                    _actor = Actor(id=id,
                                   name=name,
                                   gender=gender,
                                   country=country,
                                   birthday=birthday,
                                   deathday=deathday,
                                   shows=shows,
                                   lastUpdate=lastUpdate)
                    db.session.add(_actor)
                    db.session.commit()
                else:
                    pass

    # 先初始化一部分数据进sqlite中，采用线程池以加快速度
    response = requests.get(url='https://api.tvmaze.com/people', timeout=60)
    peopleDict = json.loads(response.text)
    peopleDict = peopleDict[120:140]  # 取20条数据即可
    peopleNameList = [people["name"] for people in peopleDict]
    executor = ThreadPoolExecutor(max_workers=20)
    all_task = [executor.submit(addInitActor, (name)) for name in peopleNameList]
    wait(all_task, return_when=ALL_COMPLETED)


dbInit()

# 参数解析
parser_Actors_postActor = reqparse.RequestParser()
parser_Actors_postActor.add_argument('name', type=str, help='')
parser_Actors_getActotList = reqparse.RequestParser()
parser_Actors_getActotList.add_argument('order', type=str, help='', default="+id")
parser_Actors_getActotList.add_argument('page', type=int, help='', default=1)
parser_Actors_getActotList.add_argument('size', type=int, help='', default=5)
parser_Actors_getActotList.add_argument('filter', type=str, help='', default="name,gender,country")


@api.route('/actors', )
class _Actors(Resource):
    @api.doc(params={'name': 'add actor name', })
    def post(self):
        '''add a actor'''
        # 提取入参
        args = parser_Actors_postActor.parse_args()
        name = args['name']
        rsp = None
        name = "".join([c if c.isalnum() else " " for c in name])  # 用空格替换所有非英语字符
        response = requests.get(url='https://api.tvmaze.com/search/people?q={0}'.format(name),
                                timeout=60)
        actorList = json.loads(response.text)
        for actor in actorList:
            # 忽略大小写、以及除字母、空格、数字之外的任何字符
            actorName = str.lower(actor['person']['name'])
            actorName = str(re.sub(r'[^\w\s\d]+', ' ', actorName)).strip()
            if (str.lower(name) == actorName):
                id = actor['person']['id']
                name = actor['person']['name']
                gender = actor['person']['gender']
                if (actor['person']['country'] is not None):
                    country = actor['person']['country']['name']
                else:
                    country = None
                birthday = actor['person']['birthday']
                if (birthday is not None):
                    birthdayItemList = str(birthday).split('-')
                    birthday = '{0}-{1}-{2}'.format(birthdayItemList[2], birthdayItemList[1], birthdayItemList[0])
                deathday = actor['person']['deathday']
                if (deathday is not None):
                    deathdayItemList = str(deathday).split('-')
                    deathday = '{0}-{1}-{2}'.format(deathdayItemList[2], deathdayItemList[1], deathdayItemList[0])
                response = requests.get(url='https://api.tvmaze.com/people/{0}/castcredits?embed=show'.format(id),
                                        timeout=60)
                shows = ','.join(
                    ['show {0}'.format(item['_embedded']['show']['id']) for item in json.loads(response.text)])
                lastUpdate = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                # 无则新增，有则更新
                if (Actor.query.filter_by(id=id).first() == None):
                    _actor = Actor(id=id,
                                   name=name,
                                   gender=gender,
                                   country=country,
                                   birthday=birthday,
                                   deathday=deathday,
                                   shows=shows,
                                   lastUpdate=lastUpdate)
                    db.session.add(_actor)
                    db.session.commit()
                else:
                    actorUpdate = Actor.query.filter_by(id=id).update(dict(
                        id=id,
                        name=name,
                        gender=gender,
                        country=country,
                        birthday=birthday,
                        deathday=deathday,
                        shows=shows,
                        lastUpdate=lastUpdate
                    ))
                    db.session.commit()
                rsp = {
                    "id": id,
                    "last-update": lastUpdate,
                    "_links": {
                        "self": {
                            "href": request.url + '/{0}'.format(id)
                        }
                    }
                }
                break
        return make_response(jsonify(rsp), 201)

    @api.doc(params={'order': 'sort order',
                     'page': 'page num',
                     'size': 'page size',
                     'filter': 'filter fields'})
    def get(self):
        '''get actor list'''
        rsp = None
        args = parser_Actors_getActotList.parse_args()
        order = args['order'].replace(' ', '+')  # +号会出现URL转义，会出现get请求传值为空的情况，需要把参数中的空格替换成原本的符号+
        page = args['page']
        size = args['size']
        filter = args['filter']
        orderList = []
        for orderItem in order.split(','):
            seq = orderItem[0]
            field = orderItem[1:]
            orderList.append(eval("Actor.{0}.asc()".format(field))
                             if seq == '+' else
                             eval("Actor.{0}.desc()".format(field)))
        fieldList = []
        for field in filter.split(','):
            if (field == 'last-update'): field = 'lastUpdate'  # 注意此处做个转换
            fieldList.append(eval("{0}.{1}".format('Actor', field)))
        actorDetailList = Actor.query.order_by(*orderList).with_entities(*fieldList).paginate(page=int(page),
                                                                                              per_page=int(size),
                                                                                              error_out=True).items
        actorList = []
        for actor in actorDetailList:
            tmpdict = dict()
            fields = actor._fields
            for field in fields:
                if (field == 'lastUpdate'):
                    tmpdict['last-update'] = eval("actor.{0}".format(field))  # 注意此处做个转换
                else:
                    tmpdict[field] = eval("actor.{0}".format(field))
            actorList.append(tmpdict)
        urlItemList = request.url.split('&')
        for index in range(len(urlItemList)):
            if urlItemList[index].find("page=") != -1:
                item_list = urlItemList[index].split("=")
                item_list[1] = int(item_list[1]) + 1
                urlItemList[index] = item_list[0] + '=' + str(item_list[1])
        nextHref = "&".join([str(item) for item in urlItemList])
        rsp = {
            "page": page,
            "page-size": size,
            "actors": [*actorList],
            "_links": {
                "self": {"href": request.url},
                "next": {"href": nextHref}
            }
        }
        return make_response(jsonify(rsp), 200)


parser_Actors_actorPatch = reqparse.RequestParser()
parser_Actors_actorPatch.add_argument('name', type=str, help='')
parser_Actors_actorPatch.add_argument('gender', type=str, help='')
parser_Actors_actorPatch.add_argument('country', type=str, help='')
parser_Actors_actorPatch.add_argument('birthday', type=str, help='')
parser_Actors_actorPatch.add_argument('deathday', type=str, help='')
parser_Actors_actorPatch.add_argument('shows', type=str, help='')


@api.route('/actors/<int:id>')
class _Actor(Resource):
    def get(self, id):
        '''get a actor'''
        # actorDetail = Actor.query.filter_by(id=id).all() #return a actor list
        actorDetail = Actor.query.get_or_404(id)
        previousHref = request.url.split('/')
        previousHref[-1] = int(previousHref[-1]) - 1
        previousHref = "/".join([str(item) for item in previousHref])
        nextHref = request.url.split('/')
        nextHref[-1] = int(nextHref[-1]) + 1
        nextHref = "/".join([str(item) for item in nextHref])
        rsp = {"id": actorDetail.id,
               "last-update": actorDetail.lastUpdate,
               "name": actorDetail.name,
               "country": actorDetail.country,
               "birthday": actorDetail.birthday,
               "deathday": actorDetail.deathday,
               "shows": actorDetail.shows.split(','),
               "_links": {
                   "self": {"href": request.url},
                   "previous": {"href": previousHref},
                   "next": {"href": nextHref},
               }
               }
        return make_response(jsonify(rsp), 200)

    def delete(self, id):
        '''delete a actor'''
        delActor = Actor.query.get_or_404(id)  # inspect delActor if exist
        Actor.query.filter_by(id=id).delete()
        db.session.commit()
        rsp = {
            "message": "The actor with id {0} was removed from the database!".format(id),
            "id": id
        }
        return make_response(jsonify(rsp), 200)

    @api.doc(params={'name': 'actor name',
                     'gender': 'actor gender',
                     'country': 'actor country',
                     'birthday': 'actor birthday',
                     'deathday': 'actor deathday',
                     'shows': 'actor shows'})
    def patch(self, id):
        '''update a actor'''
        patchActor = Actor.query.get_or_404(id)  # inspect patchActor if exist
        # request_params = request.get_json() #不建议使用该方式
        request_params = parser_Actors_actorPatch.parse_args()
        # del item having null value
        for key in list(request_params.keys()):
            if not request_params.get(key):
                request_params.pop(key)
        lastUpdate = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        request_params['lastUpdate'] = lastUpdate
        actorUpdate = Actor.query.filter_by(id=id).update(request_params)
        db.session.commit()
        rsp = {"id": id,
               "last-update": lastUpdate,
               "_links": {
                   "self": {"href": request.url},
               }
               }
        return make_response(jsonify(rsp), 200)


parser_Actors_actorStatic = reqparse.RequestParser()
parser_Actors_actorStatic.add_argument('format', type=str, help='')
parser_Actors_actorStatic.add_argument('by', type=str, help='')


@api.route('/actors/statistics')
class _ActorsStatistic(Resource):
    @api.doc(params={'format': 'json or image format',
                     'by': 'group by fields', })
    def get(self):
        '''actors statistic'''
        rsp = dict()
        args = parser_Actors_actorStatic.parse_args()
        format = args['format']
        by = args['by'].split(',')
        rsp['total'] = Actor.query.count()  # 总数据数
        rsp['total-updated'] = int(pd.Series(Actor.query.with_entities(Actor.lastUpdate).all()).apply(
            lambda x: datetime.now() - datetime.strptime(x[0], '%Y-%m-%d-%H:%M:%S') < timedelta(seconds=86400)
        ).value_counts().iloc[[True]].values[0])  # 24小时内更新过的数据数
        for byItem in by:
            if (byItem != 'life_status'):
                statistic = Actor.query.with_entities(
                    eval("Actor.{0}".format(byItem)),
                    func.count('*'),
                ).group_by(eval("Actor.{0}".format(byItem))).all()
                statisticDict = dict()
                cumCount = 0
                for item in statistic:
                    cumCount += item[1]
                    statisticDict[item[0]] = item[1]
                statisticDict = {k: round(v / cumCount * 100, 1) for k, v in statisticDict.items()}
                rsp['by-{0}'.format(byItem)] = statisticDict
            else:
                field = 'deathday'
                statistic = Actor.query.with_entities(
                    eval("Actor.{0}".format(field)),
                    func.count('*'),
                ).group_by(eval("Actor.{0}".format(field))).all()
                statisticDict = dict()
                cumCount = 0
                for item in statistic:
                    cumCount += item[1]
                    statisticDict[item[0]] = item[1]
                reprocessStatisticDict = dict()
                statisticDict = {k: round(v / cumCount * 100, 1) for k, v in statisticDict.items()}
                reprocessStatisticDict['deathed'] = 100 - statisticDict[None]
                reprocessStatisticDict['living'] = statisticDict[None]
                rsp['by-{0}'.format(byItem)] = reprocessStatisticDict
        if format == 'json':
            return make_response(jsonify(rsp), 200)
        else:
            fig, axes = plt.subplots(len(by), 1)  # 创建画布
            needPlotItemDict = {}
            for key in rsp:
                if 'by' in key:
                    needPlotItemDict[key.split('-')[1]] = rsp[key]
            if (len(needPlotItemDict) > 1):
                cnt = 0
                for k, v in needPlotItemDict.items():
                    labels = [str(key) for key in v.keys()]
                    values = list(v.values())
                    colors = ['red', 'lightskyblue']
                    axes[cnt].pie(values,
                                  colors=colors,
                                  labels=labels,
                                  explode=None,
                                  autopct='%.1f%%',  # 数值设置为保留固定小数位的百分数
                                  shadow=False,  # 无阴影设置
                                  startangle=90,  # 逆时针起始角度设置
                                  pctdistance=0.1,  # 数值距圆心半径背书距离
                                  labeldistance=3  # 图例距圆心半径倍距离
                                  )
                    axes[cnt].axis('equal')  # x,y轴刻度一致，保证饼图为圆形
                    axes[cnt].legend(loc='best')
                    axes[cnt].set_title('{0} distrubute figure'.format(k))
                    cnt += 1
            else:
                k = list(needPlotItemDict.items())[0][0]
                v = list(needPlotItemDict.items())[0][1]
                labels = [str(key) for key in v.keys()]
                values = list(v.values())
                colors = ['red', 'lightskyblue']
                axes.pie(values,
                         colors=colors,
                         labels=labels,
                         explode=None,
                         autopct='%.1f%%',  # 数值设置为保留固定小数位的百分数
                         shadow=False,  # 无阴影设置
                         startangle=90,  # 逆时针起始角度设置
                         pctdistance=0.1,  # 数值距圆心半径背书距离
                         labeldistance=3  # 图例距圆心半径倍距离
                         )
                axes.axis('equal')  # x,y轴刻度一致，保证饼图为圆形
                axes.legend(loc='best')
                axes.set_title('{0} distrubute figure'.format(k))
            # fig.set_tight_layout(True)
            # fig.savefig('figure.jpg', dpi=600)  # 将饼图保存到本地，格式为png格式，每英寸点数分辨率设置为600
            output = io.BytesIO()
            FigureCanvas(fig).print_png(output)
            return Response(output.getvalue(), mimetype='image/jpg')


'''
    可在控制台环境下进行调试，进入方式依次为:
    export FLASK_APP = actorApi.py
    flask shell
    之后可操纵db数据库对象及Actor模型类对象
'''


@app.shell_context_processor
def make_shell_context():
    return {
        'db': db,
        'Actor': Actor
    }


if __name__ == '__main__':
    app.run(debug=True)

'''
参考文献:
    1、https://zhuanlan.zhihu.com/p/361393023[flask之Restful_API的完美使用]
    2、Create A REST API with Python Flask and Flask-RestX -Project Tutorial[https://www.youtube.com/watch?v=-XuS0cfkvuA&t=737s]
'''
