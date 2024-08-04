'''
Code for obtaining Fishtank Learning problems.

Author: Lucy Li (@lucy3)
Email:  lucyl@allenai.org
'''
from bs4 import BeautifulSoup
import urllib.request
import os
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import csv
from utils import *
import requests

ROOT = '../'
DATA = os.path.join(ROOT, 'Fishtank_Learning')

def get_unit_links(): 
    root_url = 'https://www.fishtanklearning.org/curriculum/math/'
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    source = urllib.request.urlopen(root_url)
    html = source.read().decode()
    with open(os.path.join(DATA, 'units', 'unit_landing_page.html'), 'w') as outfile: 
        outfile.write(html + '\n')

def scrape_unit_pages(): 
    root_url = 'https://www.fishtanklearning.org'
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    with open(os.path.join(DATA, 'units', 'unit_landing_page.html'), 'r') as infile: 
        html = infile.read()
        soup = BeautifulSoup(html, features='lxml')
        courses = soup.find("div", {"id": "courses"})
        rows = courses.find_all("div", {"class": 'mx-4'})
        for row in tqdm(rows): 
            href = row.find('a')['href']
            parts = href.split('/')
            name = parts[3] + '_' + parts[4] + '.html'
            url = root_url + href
            source = urllib.request.urlopen(url)
            html = source.read().decode()
            with open(os.path.join(DATA, 'units', name), 'w') as outfile: 
                outfile.write(html + '\n') 

def scrape_lesson_pages(): 
    unit_folder = os.path.join(DATA, 'units')
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    for f in tqdm(os.listdir(unit_folder)): 
        if f == 'unit_landing_page.html': continue
        with open(os.path.join(unit_folder, f), 'r') as infile: 
            html = infile.read()
            soup = BeautifulSoup(html, features='lxml')
            lessons = soup.find('div', {'id': 'lesson_map'})
            numbers = lessons.find_all("span", {'class': 'lesson_map__number'})
            numbers = [int(n.text) for n in numbers]
            max_lesson = max(numbers)
            root_url = 'https://www.fishtanklearning.org/curriculum/math/' + f.replace('_', '/').replace('.html', '') + '/'
            assert numbers == list(range(1, max_lesson + 1))
            for num in numbers:
                url = root_url + 'lesson-' + str(num) + '/'
                name = f.replace('.html', '') + '_' + 'lesson-' + str(num) + '.html'
                source = urllib.request.urlopen(url)
                html = source.read().decode()
                with open(os.path.join(DATA, 'lessons', 'html', name), 'w') as outfile: 
                    outfile.write(html + '\n') 
            
def parse_lessons(): 
    lesson_folder = os.path.join(DATA, 'lessons', 'html')
    all_headers = defaultdict(list)
    for f in tqdm(os.listdir(lesson_folder)): 
        with open(os.path.join(lesson_folder, f), 'r') as infile: 
            html = infile.read()
            soup = BeautifulSoup(html, features="lxml")
            standards = soup.find('div', {'id': 'standards'})
            standard_dict = defaultdict(list)
            curr_key = None
            lesson_components = {}
            if not standards: 
                print("No standards found:", f)
            else: 
                for child in standards.findChildren(): 
                    if child.name == 'h3' and 'Standards' in child.text: 
                        curr_key = child.text.strip()
                    if child.name == 'span' and child.has_attr('class') and 'short-code' in child['class']:  
                        standard_dict[curr_key].append(child.text.strip())
            lesson_components['standards'] = standard_dict
            anchor_problems = soup.find('div', {'id': 'anchor_problems'})
            if anchor_problems: 
                lesson_components['anchor_problems'] = anchor_problems.prettify()
            anchor_tasks = soup.find('div', {'id': 'anchor_tasks'})
            if anchor_tasks: 
                lesson_components['anchor_task'] = anchor_tasks.prettify()
            target_task = soup.find('div', {'id': 'target_task'})
            if target_task: 
                lesson_components['target_task'] = target_task.prettify()
            with open(os.path.join(DATA, 'lessons', 'v0', f.replace('.html', '.json')), 'w') as outfile: 
                json.dump(lesson_components, outfile)

def get_text_and_elements(html, global_id, image_folder): 
    refs = html.find_all("div", {"class": 'references'})
    for ref in refs: 
        ref.replaceWith('')
    refs = html.find_all("div", {"class": 'reference'})
    for ref in refs: 
        ref.replaceWith('')
    assert "References" not in html.text

    extra = html.find_all("div", {"class": 'bg-math-teal-opaque'})
    for e in extra: 
        #print("TEAL", e.text.strip())
        e.replaceWith('')

    extra = html.find_all("h4")
    for e in extra: 
        #print(e.text.strip())
        e.replaceWith('') 

    extra = html.find_all("div", {"class": 'pt-8'})
    for e in extra: 
        #print("PT-8", e.text.strip())
        e.replaceWith('')

    header = html.find_all("div", {"class": "math"})
    for h in header:
        #print("HEADER", h.text.strip()) 
        h.replaceWith('')
        
    elements = defaultdict(dict)
    text_elements = set(['table', 'img'])
    tables = html.find_all('table')
    for i, table in enumerate(tables):
        elements["###TABLE" + str(i) + "###"] = str(table)
        table.replaceWith("\n###TABLE" + str(i) + "###\n")
    images = html.find_all('img')
    for i, image in enumerate(images): 
        image_url = image['src']
        response = requests.get(image_url)
        img_data = response.content
        extension = response.headers['Content-Type'].split('/')[-1]
        if extension == 'svg+xml': 
            extension = 'svg'
        image_filename = global_id + '_IMAGE' + str(i) + '.' + extension
        with open(os.path.join(image_folder, image_filename), 'wb') as handler: 
            handler.write(img_data) 
        elements["###IMAGE" + str(i) + "###"] = image_filename
        image.replaceWith("\n###IMAGE" + str(i) + "###\n")
    html_text = html.text.split('\n')
    text = ''
    for line in html_text: 
        if line.strip() != '': 
            text += line + '\n'
    return text, elements

def lesson_cleanup(): 
    '''
    - id: in format "fl_problem_#"
    - metadata
        - problem_activity_type: always 'problem_task'
        - url: of page we scraped from
        - html: name of html file containing downloaded page
        - grade or subject area
        - unit: title
        - lesson: number
        - problem_activity_html: html of problem
    - text
    - elements
    - standards
    - acquisition_date: '2024-03-05'
    - source: "Fishtank Learning"
    '''
    lesson_folder = os.path.join(DATA, 'lessons', 'v0')
    outfile = open(os.path.join(DATA, 'v1', 'lessons.jsonl'), 'w')
    idx = 0
    for f in tqdm(sorted(os.listdir(lesson_folder))): 
        with open(os.path.join(lesson_folder, f), 'r') as infile: 
            d = json.load(infile)
            problems = [p for p in d.keys() if p != 'standards']
            standards_list = []
            for k in d['standards']: 
                assert k in set(['Foundational Standards', 'Core Standards'])
                if k == 'Foundational Standards': 
                    relation = 'Building On'
                elif k == 'Core Standards': 
                    relation = 'Addressing'
                for s in d['standards'][k]: 
                    standards_list.append([relation, s])

            ret = {}
            for p in problems: 
                problem_activity_html = BeautifulSoup(d[p], features="lxml")

                global_id = 'fl_problem_' + str(idx).rjust(6, '0')
                ret['id'] = global_id
                ret['metadata'] = {}
                ret['metadata']['problem_activity_type'] = p
                ret['metadata']['url'] = 'https://www.fishtanklearning.org/curriculum/math/' + f.replace('_', '/').replace('.json', '') + '/' 
                ret['metadata']['html'] = f.replace('.json', '.html')
                parts = f.split('_')
                ret['metadata']['grade / subject'] = parts[0]
                ret['metadata']['unit'] = parts[1]
                ret['metadata']['lesson_number'] = int(parts[2].replace('lesson-', '').replace('.json', ''))
                ret['metadata']['problem_activity_html'] = d[p]

                image_folder = os.path.join(DATA, 'lessons/images')
                os.makedirs(image_folder, exist_ok=True)

                text, elements = get_text_and_elements(problem_activity_html, global_id, \
                                                       image_folder)

                ret['text'] = text
                ret['elements'] = elements

                ret['standards'] = standards_list
                ret['acquisition_date'] = '2024-03-05'
                ret['source'] = 'Fishtank Learning'

                outfile.write(json.dumps(ret) + '\n')

                idx += 1
    outfile.close()

def get_lessons(): 
    #get_unit_links()
    #scrape_unit_pages()
    #scrape_lesson_pages()
    #parse_lessons()
    lesson_cleanup()

get_lessons()