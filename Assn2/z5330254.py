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
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///z5330254.db'
db = SQLAlchemy(app)
api = Api(app, version='1.0', title='API Doc', description='', doc='/')


class Actor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    lastUpdate = db.Column(db.String(255), nullable=False)
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
        name = "".join([c if c.isalnum() else " " for c in name])
        response = requests.get(url='https://api.tvmaze.com/search/people?q={0}'.format(name),
                                timeout=60)

        actorList = json.loads(response.text)
        for actor in actorList:
            # Ignore case, and any characters except letters, spaces, and numbers
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
                # If not, add and update
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

    # First initialize some data into SQLite
    response = requests.get(url='https://api.tvmaze.com/people', timeout=60)
    peopleDict = json.loads(response.text)
    peopleDict = peopleDict[120:140]  # ???20???????????????
    peopleNameList = [people["name"] for people in peopleDict]
    executor = ThreadPoolExecutor(max_workers=20)
    all_task = [executor.submit(addInitActor, (name)) for name in peopleNameList]
    wait(all_task, return_when=ALL_COMPLETED)


dbInit()

# Parameter analysis
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
        args = parser_Actors_postActor.parse_args()
        name = args['name']
        rsp = None
        name = "".join([c if c.isalnum() else " " for c in name])
        response = requests.get(url='https://api.tvmaze.com/search/people?q={0}'.format(name),
                                timeout=60)
        actorList = json.loads(response.text)
        for actor in actorList:
            # Ignore case, and any characters except letters, spaces, and numbers
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
                # If not, add and update
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
        order = args['order'].replace(' ', '+')
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
            if (field == 'last-update'): field = 'lastUpdate'  # ????????????????????????
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
                    tmpdict['last-update'] = eval("actor.{0}".format(field))  # ????????????????????????
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
        # total number
        rsp['total'] = Actor.query.count()
        rsp['total-updated'] = int(pd.Series(Actor.query.with_entities(Actor.lastUpdate).all()).apply(
            lambda x: datetime.now() - datetime.strptime(x[0], '%Y-%m-%d-%H:%M:%S') < timedelta(seconds=86400)
        ).value_counts().iloc[[True]].values[0])
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
            fig, axes = plt.subplots(len(by), 1)
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
                                  autopct='%.1f%%',
                                  shadow=False,
                                  startangle=90,
                                  pctdistance=0.1,
                                  labeldistance=3
                                  )
                    axes[cnt].axis('equal')
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
                         autopct='%.1f%%',
                         shadow=False,
                         startangle=90,
                         pctdistance=0.1,
                         labeldistance=3
                         )
                axes.axis('equal')
                axes.legend(loc='best')
                axes.set_title('{0} distrubute figure'.format(k))
            # fig.set_tight_layout(True)
            # fig.savefig('figure.jpg', dpi=600)
            output = io.BytesIO()
            FigureCanvas(fig).print_png(output)
            return Response(output.getvalue(), mimetype='image/jpg')


'''
    run method:
    export FLASK_APP = z5330254.py
    flask shell
'''


@app.shell_context_processor
def make_shell_context():
    return {
        'db': db,
        'Actor': Actor
    }


if __name__ == '__main__':
    app.run(debug=True)
