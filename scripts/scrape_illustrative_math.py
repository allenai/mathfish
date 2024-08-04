'''
Code for obtaining Illustrative Math problems from
various parts of their curriculum. 

Author: Lucy Li (@lucy3)
Email:  lucyl@allenai.org
'''
from bs4 import BeautifulSoup
from urllib.request import urlopen
import os
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import csv
from utils import *
import requests

ROOT = '../'
DATA = os.path.join(ROOT, 'Illustrative_Math')
        
def scrape_grade_task_page(root_url): 
    grades = ['K', '1', '2', '3', '4', '5', '6', '7', '8', 'HSN', 'HSA', 'HSF', 'HSG', 'HSS']
    for grd in grades: 
        source = urlopen(root_url + grd)
        html = source.read().decode()
        with open(os.path.join(DATA, 'tasks/html', grd) + '.html', 'w') as outfile: 
            outfile.write(html + '\n')
            
def scrape_all_task_links(root_url):
    grades = ['K', '1', '2', '3', '4', '5', '6', '7', '8', 'HSN', 'HSA', 'HSF', 'HSG', 'HSS']
    for grd in tqdm(grades): 
        with open(os.path.join(DATA, 'tasks/html', grd) + '.html', 'r') as infile: 
            html = infile.read()
            soup = BeautifulSoup(html, features="lxml")
            for a in soup.find_all('a', href=True):
                if 'content-standards' in a['href'] and 'tasks' in a['href']: 
                    source = urlopen(root_url + a['href'])
                    task_html = source.read().decode()
                    filename = a['href'].replace('/', '_')
                    with open(os.path.join(DATA, 'tasks/html', filename), 'w') as outfile: 
                        outfile.write(task_html + '\n')

def scrape_tasks(): 
    '''
    https://tasks.illustrativemathematics.org/content-standards
    Scrape date: 2024-02-05
    '''
    root_url = 'https://tasks.illustrativemathematics.org/'
    scrape_grade_task_page(root_url)
    scrape_all_task_links(root_url)
    
def get_text_and_elements(html, global_id, image_folder, problem_activity_type): 
    if problem_activity_type in ['lesson', 'practice', 'modeling prompt']: 
        # remove span that is embedded-modal-dialog
        embedded = html.find_all('span', {'class': 'embedded-modal-dialog'})
        for embed in embedded: 
            embed.replaceWith('')
    
    elements = defaultdict(dict)
    # there is also an element called "figure" but that is a parent of a collection of img
    text_elements = set(['table', 'img'])
    tables = html.find_all('table')
    for i, table in enumerate(tables):
        elements["###TABLE" + str(i) + "###"] = str(table)
        table.replaceWith("\n###TABLE" + str(i) + "###\n")
    images = html.find_all('img')

    for i, image in enumerate(images): 
        image_url = image['src']
        if (problem_activity_type in ['lesson', 'practice', 'modeling prompt']) and 'expand-' in image_url: 
            continue # this is the expansion arrows on the page
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
    
def parse_task_helper(f, html, math_standards, global_id): 
    '''
    Transform task page html into standard json shared across sources
    - id: in format "im_task_#"
    - metadata
        - problem_activity_type: always 'task' here
        - url: of page we scraped from
        - html: name of html file containing downloaded page
        - title: title of task
        - problem_activity_html: html of problem or activity only
    - text
    - elements
    - standards
    - acquisition_date
    - source: always "Illustrative Mathematics" here
    '''
    ret = {}
    # global ID
    global_id = 'im_task_' + str(global_id).rjust(6, '0')
    ret['id'] = global_id
    
    soup = BeautifulSoup(html, features="lxml")
        
    # source-specific metadata
    ret['metadata'] = {}
    ret['metadata']['problem_activity_type'] = 'task'
    url = 'https://tasks.illustrativemathematics.org/' + f.replace('.html', '').replace('_', '/')
    ret['metadata']['url'] = url 
    ret['metadata']['html'] = f
    header = soup.find_all("h1")
    assert len(header) == 1
    ret['metadata']['title'] = header[0].text
    
    # task
    details = soup.find_all("div", {"class": "detail"})
    problem_activity_html = ''
    for detail in details: 
        subtitle = detail.find_all("h2")
        if subtitle and subtitle[0].text == 'Task': 
            problem_activity_html = detail
    assert problem_activity_html
    ret['metadata']['problem_activity_html'] = problem_activity_html.prettify() # html snippet of the problem or activity
    
    text, elements = get_text_and_elements(problem_activity_html, global_id, 
                                           os.path.join(DATA, 'tasks/images'), 'task')
    
    # text
    ret['text'] = text
    
    # images
    ret['elements'] = elements # image IDs in text to filenames, or table to html, etc
    
    # standards
    alignment_div = soup.find_all("span", {"class": "alignment"})
    assert len(alignment_div) == 1
    standards = alignment_div[0].text.replace('Alignments to Content Standards:', '').strip().split('\n')
    standards_list = []
    for standard in standards:
        standard = standardize_standard(standard, math_standards)
        standards_list.append(('Alignment', standard))
    ret['standards'] = standards_list
    
    # source
    ret['acquisition_date'] = '2024-02-05' # based on scrape date
    ret['source'] = 'Illustrative Mathematics'
    return ret

def parse_tasks(): 
    math_standards = get_math_standards()
    global_id = 0
    with open(os.path.join(DATA, 'v1', 'tasks.jsonl'), 'w') as outfile: 
        for f in tqdm(sorted(os.listdir(os.path.join(DATA, 'tasks/html')))): 
            if 'content-standards' in f: 
                with open(os.path.join(DATA, 'tasks/html', f), 'r') as infile: 
                    html = infile.read()
                    task_dict = parse_task_helper(f, html, math_standards, global_id)
                    outfile.write(json.dumps(task_dict) + '\n')
                    global_id += 1
                
def reformat_tasks_to_csv(): 
    header = ['id', 'problem_activity_type', 'url', 'html', 'title', 'text', 'elements', 'standard']
    with open(os.path.join(DATA, 'tasks.csv'), 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=header)
        writer.writeheader()
        with open(os.path.join(DATA, 'v1', 'tasks.jsonl'), 'r') as infile:
            for line in infile: 
                d = json.loads(line)
                standards_list = d['standards']
                for tup in standards_list: 
                    standard = tup[1]
                    ret = {}
                    for key in d: 
                        if key in header: 
                            ret[key] = d[key]
                    for key in d['metadata']: 
                        if key in header: 
                            ret[key] = d['metadata'][key]
                    ret['standard'] = standard
                    writer.writerow(ret)
                    
def get_tasks(): 
    scrape_tasks()
    parse_tasks()
    reformat_tasks_to_csv()
                    
def scrape_unit_pages(): 
    unit_pages = {'acc-6': 'https://im.kendallhunt.com/MS_ACC/teachers/1/index.html', 
                  'acc-7': 'https://im.kendallhunt.com/MS_ACC/teachers/2/index.html', 
                  'grade-6': 'https://im.kendallhunt.com/MS/teachers/1/index.html', 
                  'grade-7': 'https://im.kendallhunt.com/MS/teachers/2/index.html', 
                  'grade-8': 'https://im.kendallhunt.com/MS/teachers/3/index.html',
                  'algebra-1': 'https://im.kendallhunt.com/HS/teachers/1/index.html',
                  'geometry': 'https://im.kendallhunt.com/HS/teachers/2/index.html',
                  'algebra-2': 'https://im.kendallhunt.com/HS/teachers/3/index.html',
                  'algebra-1-supports': 'https://im.kendallhunt.com/HS/teachers/4/index.html'}
    root_link = 'https://im.kendallhunt.com/K5/teachers/'
    k5 = ['kindergarten', 'grade-1', 'grade-2', 'grade-3', 'grade-4', 'grade-5']
    for grade in k5:
        unit_pages[grade] = 'https://im.kendallhunt.com/K5/teachers/' + grade + '/units.html'
        
    for grade in unit_pages: 
        source = urlopen(unit_pages[grade])
        html = source.read().decode()
        with open(os.path.join(DATA, 'units', grade + '.html'), 'w') as outfile: 
            outfile.write(html + '\n')
            
def scrape_lesson_links(rescrape=False): 
    '''
    Scrape date: 2024-02-08
    '''
    lesson_list_folder = os.path.join(DATA, 'lessons', 'lesson_lists')
    root = 'https://im.kendallhunt.com/'
    for f in os.listdir(os.path.join(DATA, 'units')): 
        with open(os.path.join(DATA, 'units', f), 'r') as infile: 
            html = infile.read()
            soup = BeautifulSoup(html, features="lxml")
            divs = soup.find_all("div", {"class": "im-c-grid"})
            for div in divs: 
                for link in div.find_all('a'): 
                    name = f.replace('.html', '') + '_' + link.find_all('h3')[0].text.lower().replace(' ', '-')
                    url = root + link['href']
                    source = urlopen(url)
                    html = source.read().decode()
                    with open(os.path.join(lesson_list_folder, name + '.html'), 'w') as outfile: 
                        outfile.write(html + '\n')
                        
    for f in os.listdir(lesson_list_folder): 
        if not f.endswith('.html'): continue
        grade = f.split('_')[0]
        k5 = ['kindergarten', 'grade-1', 'grade-2', 'grade-3', 'grade-4', 'grade-5']
        with open(os.path.join(lesson_list_folder, f), 'r') as infile: 
            html = infile.read()
            soup = BeautifulSoup(html, features="lxml")
            div = soup.find_all("div", {"class": "im-c-container"})[1]
            for link in div.find_all('a'): 
                url = root + link['href']
                if 'preparation' not in url: continue
                if grade in k5: 
                    url = url.replace('preparation.html', 'lesson.html')
                else: 
                    url = url.replace('preparation.html', 'index.html')
                name = f.replace('.html', '') + '_lesson-' + link.find('span').text
                if not rescrape and os.path.exists(os.path.join(DATA, 'lessons', 'html', name + '.html')): 
                    continue
                print(url)
                source = urlopen(url)
                html = source.read().decode()
                with open(os.path.join(DATA, 'lessons', 'html', name + '.html'), 'w') as outfile: 
                    outfile.write(html + '\n')
                    
def scrape_lesson_pages(): 
    root = 'https://im.kendallhunt.com'
    i = 0
    for f in tqdm(os.listdir(os.path.join(DATA, 'lessons', 'html'))): 
        if not f.endswith('.html'): continue
        sections = []
        with open(os.path.join(DATA, 'lessons', 'html', f), 'r') as infile: 
            html = infile.read()
            soup = BeautifulSoup(html, features="lxml")
            url = root + soup.find('nav', {'class': 'im-c-tabs'}).find('a', {'class': 'active'})['href']
            body = soup.find('main', {'class': 'im-c-main'})
            curr_section = {} # current activity or activity header
            for child in body.children: 
                if not child.name == 'div': continue
                if 'im-c-hero' in child['class']: 
                    if curr_section: 
                        sections.append(curr_section)
                    curr_section = {}
                    
                    curr_section['title'] = child.find('h2').text
                    curr_section['html'] = f
                    curr_section['relation'] = defaultdict(list)
                    curr_section['url'] = url
                    for section in child.find_all('section'): 
                        if 'CCSS Standards' not in section.text: continue
                        curr_relation = ''
                        for section_child in section.children: 
                            if section_child.name == 'p': 
                                if section_child.has_attr('class') and 'im-c-hero__underline' in section_child['class']: 
                                    curr_relation = section_child.text
                            if section_child.name == 'ul': 
                                standards = [s for s in section_child.text.split('\n') if s != '']
                                curr_section['relation'][curr_relation].extend(standards)
                elif 'im-c-container--content' in child['class']: 
                    curr_section['problem_activity_html'] = child.prettify()
            if curr_section: # fence post
                sections.append(curr_section)
        with open(os.path.join(DATA, 'lessons', 'v0', f.replace('.html', '.jsonl')), 'w') as outfile:     
            for section in sections: 
                outfile.write(json.dumps(section) + '\n')
            
def lesson_cleanup(): 
    '''
    - id: in format "im_lesson_#"
    - metadata
        - problem_activity_type: always 'lesson activity' here
        - url: of page we scraped from
        - html: name of html file containing downloaded page
        - title: title of task
        - grade or subject area: 
        - unit: number
        - lesson: number
        - problem_activity_html: html of problem or activity only
    - text
    - elements
    - standards
    - acquisition_date
    - source: always "Illustrative Mathematics" here
    '''
    idx = 0
    outfile = open(os.path.join(DATA, 'v1', 'lessons.jsonl'), 'w')
    for f in tqdm(os.listdir(os.path.join(DATA, 'lessons', 'v0'))): 
        filepath = os.path.join(DATA, 'lessons', 'v0', f)
        with open(filepath, 'r') as infile: 
            for line in infile: 
                section_dict = json.loads(line)
                if 'Cool Down' in section_dict['title'] or 'Cool-down' in section_dict['title'] or \
                    'Cool-Down' in section_dict['title']: 
                    # if the section is a Cool Down activity, also continue since these require teacher login
                    continue
                ret = {}
                global_id = 'im_lesson_' + str(idx).rjust(6, '0')
                ret['id'] = global_id
                ret['metadata'] = {}
                ret['metadata']['problem_activity_type'] = 'lesson activity'
                ret['metadata']['url'] = section_dict['url']
                ret['metadata']['html'] = section_dict['html']
                ret['metadata']['title'] = section_dict['title']
                parts = section_dict['html'].replace('.html', '').split('_')
                ret['metadata']['grade / subject'] = parts[0]
                ret['metadata']['unit_number'] = parts[1].replace('unit-', '')
                ret['metadata']['lesson_number'] = parts[2].replace('lesson-', '')
                problem_activity_html = BeautifulSoup(section_dict['problem_activity_html'], features="lxml")
                ret['metadata']['problem_activity_html'] = section_dict['problem_activity_html']
                
                image_folder = os.path.join(DATA, 'lessons/images', parts[0])
                os.makedirs(image_folder, exist_ok=True)
                
                text, elements = get_text_and_elements(problem_activity_html, global_id, \
                                                       image_folder, 'lesson')
                
                ret['text'] = text
                ret['elements'] = elements
                standards_list = []
                for relation in section_dict['relation']: 
                    standards_set = set(section_dict['relation'][relation])
                    for s in standards_set: 
                        standards_list.append((relation, s))
                ret['standards'] = standards_list
                ret['acquisition_date'] = '2024-02-28'
                ret['source'] = 'Illustrative Mathematics'
                
                outfile.write(json.dumps(ret) + '\n')
                
                idx += 1
    outfile.close()
    
def scrape_lesson_prep(): 
    lesson_list_folder = os.path.join(DATA, 'lessons', 'lesson_lists')
    root = 'https://im.kendallhunt.com'
    for f in tqdm(os.listdir(lesson_list_folder)): 
        if not f.endswith('.html'): continue
        grade = f.split('_')[0]
        with open(os.path.join(lesson_list_folder, f), 'r') as infile: 
            html = infile.read()
            soup = BeautifulSoup(html, features="lxml")
            div = soup.find_all("div", {"class": "im-c-container"})[1]
            for link in div.find_all('a'): 
                url = root + link['href']
                if 'preparation' not in url: continue
                name = f.replace('.html', '') + '_lesson-' + link.find('span').text
                source = urlopen(url)
                html = source.read().decode()
                soup = BeautifulSoup(html, features="lxml")
                embedded = soup.find_all('span', {'class': 'embedded-modal-dialog'})
                for embed in embedded: 
                    embed.replaceWith('')
                div = soup.find('main', {'class':'im-c-main'})
                images = div.find_all('img')
                for i, image in enumerate(images): 
                    image_url = image['src']
                    if 'expand-' in image_url or 'im_certified_badge' in image_url: 
                        continue # this is the expansion arrows on the page
                    response = requests.get(image_url)
                    img_data = response.content
                    extension = response.headers['Content-Type'].split('/')[-1]
                    if extension == 'svg+xml': 
                        extension = 'svg'
                    image_filename = name + '_' + str(i) + '.' + extension
                    with open(os.path.join(DATA, 'lessons', 'lesson_prep', 'images', image_filename), 'wb') as handler:
                        handler.write(img_data)
                with open(os.path.join(DATA, 'lessons', 'lesson_prep', 'html', name + '.html'), 'w') as outfile: 
                    outfile.write(html + '\n')
                    
def get_lessons(): 
    '''
    metadata: unit, lesson, title
    '''
    #scrape_unit_pages()
    #scrape_lesson_links()
    #scrape_lesson_pages()
    lesson_cleanup()
    #scrape_lesson_prep()
    
def scrape_center_links(): 
    root = 'https://im.kendallhunt.com'
    for f in os.listdir(os.path.join(DATA, 'units')): 
        with open(os.path.join(DATA, 'units', f), 'r') as infile: 
            html = infile.read()
            soup = BeautifulSoup(html, features="lxml")
            divs = soup.find_all("nav", {"class": "im-c-tabs"})
            assert len(divs) == 1
            nav = divs[0]
            for child in nav.children: 
                if child.text.strip() == 'Centers': 
                    source = urlopen(root + child['href'])
                    center_list_html = source.read().decode()
                    filename = child['href'].split('/')[-2]
                    with open(os.path.join(DATA, 'centers/center_lists', filename), 'w') as outfile: 
                        outfile.write(center_list_html + '\n')
                        
    for f in tqdm(os.listdir(os.path.join(DATA, 'centers/center_lists'))): 
        with open(os.path.join(DATA, 'centers/center_lists', f), 'r') as infile: 
            html = infile.read()
            soup = BeautifulSoup(html, features="lxml")
            divs = soup.find_all("a", {"class": "im-c-grid__item"})
            for div in divs: 
                link = root + div['href']
                link = link.replace('overview.html', 'center.html')
                source = urlopen(link)
                center_html = source.read().decode()
                filename = f + '_' + div['href'].split('/')[-2]
                with open(os.path.join(DATA, 'centers/html', filename), 'w') as outfile: 
                    outfile.write(center_html + '\n')
                    
def scrape_center_pages(): 
    '''
    Scraped 2024-02-12
    '''
    root = 'https://im.kendallhunt.com'
    i = 0
    for f in tqdm(os.listdir(os.path.join(DATA, 'centers', 'html'))): 
        sections = []
        with open(os.path.join(DATA, 'centers', 'html', f), 'r') as infile: 
            html = infile.read()
            soup = BeautifulSoup(html, features="lxml")
            url = root + soup.find('nav', {'class': 'im-c-tabs'}).find('a', {'class': 'active'})['href']
            body = soup.find('main', {'class': 'im-c-main'})
            curr_section = {} # current activity or activity header
            for child in body.children: 
                if not child.name == 'div': continue
                if 'im-c-hero' in child['class']: 
                    if curr_section: 
                        sections.append(curr_section)
                    curr_section = {}
                    
                    curr_section['title'] = child.find('h2').text
                    curr_section['html'] = f
                    curr_section['relation'] = defaultdict(list)
                    curr_section['url'] = url
                    for section in child.find_all('section'): 
                        if 'CCSS Standards' in section.text: 
                            curr_relation = ''
                            for section_child in section.children: 
                                if section_child.name == 'p': 
                                    if section_child.has_attr('class') and 'im-c-hero__underline' in section_child['class']: 
                                        curr_relation = section_child.text
                                if section_child.name == 'ul': 
                                    standards = [s for s in section_child.text.split('\n') if s != '']
                                    curr_section['relation'][curr_relation].extend(standards)
                            section.replaceWith('<br>')
                    curr_section['problem_activity_html'] = child.prettify()
                elif 'im-c-container--content' in child['class']: 
                    curr_section['problem_activity_html'] += child.prettify()
            if curr_section: # fence post
                sections.append(curr_section)
        with open(os.path.join(DATA, 'centers', 'v0', f.replace('.html', '.jsonl')), 'w') as outfile:     
            for section in sections: 
                outfile.write(json.dumps(section) + '\n')

def center_cleanup(): 
    '''
    - id: in format "im_center_#"
    - metadata
        - problem_activity_type: always 'center' here
        - url: of page we scraped from
        - html: name of html file containing downloaded page
        - title: title of center
        - grade or subject area: 
        - problem_activity_html: html of problem or activity only
    - text
    - elements
    - standards
    - acquisition_date
    - source: always "Illustrative Mathematics" here
    '''
    idx = 0
    outfile = open(os.path.join(DATA, 'v1', 'centers.jsonl'), 'w')
    for f in tqdm(os.listdir(os.path.join(DATA, 'centers', 'v0'))): 
        filepath = os.path.join(DATA, 'centers', 'v0', f)
        with open(filepath, 'r') as infile: 
            for line in infile: 
                section_dict = json.loads(line)
                ret = {}
                global_id = 'im_center_' + str(idx).rjust(6, '0')
                ret['id'] = global_id
                ret['metadata'] = {}
                ret['metadata']['problem_activity_type'] = 'center'
                ret['metadata']['url'] = section_dict['url']
                ret['metadata']['html'] = section_dict['html']
                ret['metadata']['title'] = section_dict['title']
                parts = section_dict['html'].replace('.html', '').split('_')
                ret['metadata']['grade / subject'] = parts[0]
                problem_activity_html = BeautifulSoup(section_dict['problem_activity_html'], features="lxml")
                ret['metadata']['problem_activity_html'] = section_dict['problem_activity_html']
                
                image_folder = os.path.join(DATA, 'centers/images', parts[0])
                os.makedirs(image_folder, exist_ok=True)
                text, elements = get_text_and_elements(problem_activity_html, global_id, \
                                                       image_folder, 'center')
                
                ret['text'] = text
                ret['elements'] = elements
                standards_list = []
                for relation in section_dict['relation']: 
                    standards_set = set(section_dict['relation'][relation])
                    for s in standards_set: 
                        standards_list.append((relation, s))
                ret['standards'] = standards_list
                ret['acquisition_date'] = '2024-02-12'
                ret['source'] = 'Illustrative Mathematics'
                
                outfile.write(json.dumps(ret) + '\n')
                
                idx += 1
    outfile.close()
                
def get_centers(): 
    #scrape_center_links()
    #scrape_center_pages()
    center_cleanup()
    
def scrape_practice_pages(): 
    '''
    1) Units -> Lessons -> Practice 
    2) Units -> Practice
    '''
    root = 'https://im.kendallhunt.com'
    lesson_list_folder = os.path.join(DATA, 'lessons', 'lesson_lists')
    # Units -> Practice
    for f in os.listdir(lesson_list_folder): 
        if not f.endswith('.html'): continue
        grade = f.split('_')[0]
        with open(os.path.join(lesson_list_folder, f), 'r') as infile: 
            html = infile.read()
            soup = BeautifulSoup(html, features="lxml")
            divs = soup.find_all("nav", {"class": "im-c-tabs"})
            assert len(divs) == 1
            nav = divs[0]
            for child in nav.children: 
                if child.text.strip() == 'Practice': 
                    link = child['href']
                    source = urlopen(root + link)
                    html = source.read().decode()
                    filename = link.replace('/', '_')[1:]
                    with open(os.path.join(DATA, 'practice', 'html', filename), 'w') as outfile: 
                        outfile.write(html + '\n')
                    
    # Units -> Lessons -> Practice 
    for f in tqdm(os.listdir(os.path.join(DATA, 'lessons', 'html'))): 
        if not f.endswith('.html'): continue
        sections = []
        with open(os.path.join(DATA, 'lessons', 'html', f), 'r') as infile: 
            html = infile.read()
            soup = BeautifulSoup(html, features="lxml")
            divs = soup.find_all("nav", {"class": "im-c-tabs"})
            assert len(divs) == 1
            nav = divs[0]
            for child in nav.children: 
                if child.text.strip() == 'Practice': 
                    link = child['href']
                    source = urlopen(root + link)
                    html = source.read().decode()
                    filename = link.replace('/', '_')[1:]
                    with open(os.path.join(DATA, 'practice', 'html', filename), 'w') as outfile: 
                        outfile.write(html + '\n')
                        
def scrape_practice_problems():
    '''
    - id: in format "im_center_#"
    - metadata
        - problem_activity_type: always 'practice' here
        - url: of page we scraped from
        - html: name of html file containing downloaded page
        - grade or subject area: 
        - unit: 
        - problem_activity_html: html of problem or activity only
    - text
    - elements
    - standards
    - acquisition_date
    - source: always "Illustrative Mathematics" here
    '''
    root = 'https://im.kendallhunt.com/'
    idx = 0
    mapping = {'Algebra 2': 'algebra-2', 'Grade 2': 'grade-2', 'Geometry': 'geometry', 'Accelerated Grade 6': 'acc-6', 'Grade 7': 'grade-7', 'Grade 4': 'grade-4', 'Grade 3': 'grade-3', 'Grade 8': 'grade-8', 'Accelerated Grade 7': 'acc-7', 'Algebra 1': 'algebra-1', 'Grade 5': 'grade-5', 'Kindergarten': 'kindergarten', 'Grade 6': 'grade-6', 'Grade 1': 'grade-1'}
    s = set()
    outfile = open(os.path.join(DATA, 'v1', 'practice.jsonl'), 'w')
    for f in tqdm(os.listdir(os.path.join(DATA, 'practice', 'html'))): 
        if not f.endswith('.html'): continue
        with open(os.path.join(DATA, 'practice', 'html', f), 'r') as infile: 
            html = infile.read()
            soup = BeautifulSoup(html, features="lxml")
            div = soup.find("div", {"class": 'im-c-container--content'})
            nav = soup.find_all("button", {"class": 'im-c-dropdown__button'})
            parts = []
            for n in nav: 
                if n.text == 'Math Tools' or n.text == 'English': continue
                parts.append(n.text)
            assert len(parts) == 2
            grade_subject_area = mapping[parts[0]]
            unit = parts[1]
            problems = div.find_all("div", {'class': 'im-c-row'})
            for problem in problems: 
                title = problem.find("div", {"class": 'im-c-row__aside'}).text
                if title == 'Solution': continue
                ret = {}
                global_id = 'im_practice_' + str(idx).rjust(6, '0')
                ret['id'] = global_id
                ret['metadata'] = {}
                ret['metadata']['problem_activity_type'] = 'practice'
                ret['metadata']['url'] = root + f.replace('_', '/').replace('MC/ACC', 'MC_ACC')
                ret['metadata']['html'] = f
                ret['metadata']['grade / subject'] = grade_subject_area
                ret['metadata']['unit'] = unit
                
                paras = problem.find_all("p")
                standards_list = []
                for p in paras: 
                    if p.text.startswith("Practicing Standards:"): 
                        s_list = p.text.replace(u'\xa0', u'').replace('Practicing Standards: ', '').split(', ')
                        for s in s_list: 
                            standards_list.append(('Alignment', s))
                        p.replaceWith('')
                
                ret['metadata']['problem_activity_html'] = problem.prettify()
                
                image_folder = os.path.join(DATA, 'practice/images', grade_subject_area)
                os.makedirs(image_folder, exist_ok=True)
                text, elements = get_text_and_elements(problem, global_id, \
                                                       image_folder, 'practice')
                
                ret['text'] = text
                ret['elements'] = elements
                ret['standards'] = standards_list
                ret['acquisition_date'] = '2024-02-12'
                ret['source'] = 'Illustrative Mathematics'
                
                outfile.write(json.dumps(ret) + '\n')
                idx += 1
    outfile.close()

def get_practice(): 
    #scrape_practice_pages()
    scrape_practice_problems()

def get_modeling_prompts_list():              
    root = 'https://im.kendallhunt.com'
    list_folder = os.path.join(DATA, 'modeling_prompts', 'modeling_prompts_lists')
    for f in os.listdir(os.path.join(DATA, 'units')): 
        with open(os.path.join(DATA, 'units', f), 'r') as infile: 
            html = infile.read()
            soup = BeautifulSoup(html, features="lxml")
            divs = soup.find_all("nav", {"class": "im-c-tabs"})
            assert len(divs) == 1
            nav = divs[0]
            for child in nav.children: 
                if child.text.strip() == 'Modeling Prompts': 
                    link = child['href']
                    name = link.replace('/', '_')[1:]
                    source = urlopen(root + link)
                    html = source.read().decode()
                    with open(os.path.join(list_folder, name), 'w') as outfile: 
                        outfile.write(html + '\n')

def scrape_modeling_prompt_pages(): 
    '''
    Scraped on 02-15-2024
    '''
    root = 'https://im.kendallhunt.com'
    list_folder = os.path.join(DATA, 'modeling_prompts', 'modeling_prompts_lists')
    for f in os.listdir(list_folder): 
        with open(os.path.join(list_folder, f), 'r') as infile: 
            html = infile.read()
            soup = BeautifulSoup(html, features="lxml")
            divs = soup.find_all("div", {"class": "im-c-grid"})
            for div in divs: 
                for link in div.find_all('a'): 
                    name = f.replace('.html', '') + '_' + link.find_all('h3')[0].text.lower().replace(' ', '-')
                    url = root + link['href']
                    source = urlopen(url)
                    html = source.read().decode()
                    filename = link['href'].replace('/', '_')[1:]
                    with open(os.path.join(DATA, 'modeling_prompts', 'modeling_prompts_prep', 'html', filename), 'w') as outfile: 
                        outfile.write(html + '\n')
                        
                    task_url = url.replace('preparation.html', 'modeling_tasks.html')
                    source = urlopen(task_url)
                    html = source.read().decode()
                    filename = filename.replace('preparation.html', 'modeling_tasks.html')
                    with open(os.path.join(DATA, 'modeling_prompts', 'html', filename), 'w') as outfile: 
                        outfile.write(html + '\n')
                        
def scrape_modeling_prompts(): 
    '''
    Each modeling prompt consists of the following: 
    - In Class Launch
    - Task Statements, each consisting of Teacher Instructions, Student-Facing Statement
    '''
    prep_folder = os.path.join(DATA, 'modeling_prompts', 'modeling_prompts_prep', 'html')
    task_folder = os.path.join(DATA, 'modeling_prompts', 'html')
    outfile = open(os.path.join(DATA, 'modeling_prompts', 'v0', 'modeling_prompts_v0.jsonl'), 'w')
    for f in tqdm(os.listdir(prep_folder)): 
        ret = {}
        ret['problem_activity_html'] = ''
        ret['standards'] = defaultdict(list)
        ret['filename'] = f.replace('preparation.html', 'modeling_tasks.html')
        # get stuff from modeling prep page
        with open(os.path.join(prep_folder, f), 'r') as infile: 
            html = infile.read()
            soup = BeautifulSoup(html, features="lxml")
            embedded = soup.find_all('span', {'class': 'embedded-modal-dialog'})
            for embed in embedded: 
                embed.replaceWith('')
                
            rows = soup.find_all('div', {'class':'im-c-row'})
            for row in rows: 
                if row.text.strip().split('\n')[0] == 'In Class Launch': 
                    ret['problem_activity_html'] += row.prettify()
                    
                if row.text.strip().split('\n')[0] == 'Alignments': 
                    body = row.find('div', {'class': 'im-c-row__body'})
                    curr_relation = ''
                    
                    for child in body.children:
                        if child.text.strip() == '': continue
                        if child.name == 'h4': 
                            curr_relation = child.text.strip()
                        elif child.name == 'ul': 
                            ret['standards'][curr_relation] = child.text.strip().split('\n')
        
        # get stuff from modeling task page
        with open(os.path.join(task_folder, f.replace('preparation.html', 'modeling_tasks.html')), 'r') as infile: 
            html = infile.read()
            soup = BeautifulSoup(html, features="lxml")
            embedded = soup.find_all('span', {'class': 'embedded-modal-dialog'})
            for embed in embedded: 
                embed.replaceWith('')
                
            rows = soup.find_all('div', {'class':'im-c-row'})
            for row in rows: 
                header = row.text.strip().split('\n')[0]
                if header == 'Teacher Instructions' or header == 'Student-Facing Statement': 
                    ret['problem_activity_html'] += row.prettify()
        outfile.write(json.dumps(ret) + '\n')
    outfile.close()
        
def cleanup_modeling_prompts(): 
    '''
    - id: in format "im_modelingprompt_#"
    - metadata
        - problem_activity_type: always 'modeling prompt' here
        - url: [of pages we scraped from]
        - html: [name of html files containing downloaded pages]
        - grade or subject area: 
        - problem_activity_html: html of problem or activity only
    - text
    - elements
    - standards
    - acquisition_date
    - source: always "Illustrative Mathematics" here
    '''
    root = 'https://im.kendallhunt.com/'
    idx = 0
    outfile = open(os.path.join(DATA, 'v1', 'modeling_prompts.jsonl'), 'w')
    with open(os.path.join(DATA, 'modeling_prompts', 'v0', 'modeling_prompts_v0.jsonl'), 'r') as infile: 
        for line in tqdm(infile): 
            d = json.loads(line)
            ret = {}
            global_id = 'im_modelingprompt_' + str(idx).rjust(6, '0')
            ret['id'] = global_id
            ret['metadata'] = {}
            ret['metadata']['problem_activity_type'] = 'modeling prompt'
            f = d['filename']
            suffix = f.replace('_', '/').replace('modeling/prompts', 'modeling_prompts').replace('modeling/tasks', 'modeling_tasks')
            ret['metadata']['url'] = [root + suffix, root + suffix.replace('modeling_tasks', 'preparation')]
            ret['metadata']['html'] = [f, f.replace('modeling_tasks', 'preparation')]
            grade_subject_area = ''
            if 'teachers_3' in f: 
                grade_subject_area = 'algebra-2'
            if 'teachers_2' in f: 
                grade_subject_area = 'geometry'
            if 'teachers_1' in f: 
                grade_subject_area = 'algebra-1'
            ret['metadata']['grade / subject'] = grade_subject_area

            ret['metadata']['problem_activity_html'] = d['problem_activity_html']

            image_folder = os.path.join(DATA, 'modeling_prompts/images')
            os.makedirs(image_folder, exist_ok=True)
            
            problem = BeautifulSoup(d['problem_activity_html'], features="lxml")
            text, elements = get_text_and_elements(problem, global_id, \
                                                   image_folder, 'modeling prompt')

            ret['text'] = text
            ret['elements'] = elements
            standards_list = []
            for relation in d['standards']: 
                for s in d['standards'][relation]: 
                    standards_list.append((relation, s))
            ret['standards'] = standards_list
            ret['acquisition_date'] = '2024-02-15'
            ret['source'] = 'Illustrative Mathematics'

            outfile.write(json.dumps(ret) + '\n')
            idx += 1
    outfile.close()

def get_modeling_prompts(): 
    #get_modeling_prompts_list()
    #scrape_modeling_prompt_pages()
    #scrape_modeling_prompts()
    cleanup_modeling_prompts()
                
get_lessons()