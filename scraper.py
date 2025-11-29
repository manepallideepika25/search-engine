import json
import requests

def scrape_leetcode():
    url = "https://leetcode.com/api/problems/all/"
    response = requests.get(url)
    data = response.json()

    problems = []
    for problem in data['stat_status_pairs']:
        if problem['paid_only']:
            continue

        stat = problem['stat']
        title = stat['question__title']
        title_slug = stat['question__title_slug']
        
        problems.append({
            "title": title,
            "description": f"Difficulty: {problem['difficulty']['level']}",
            "url": f"https://leetcode.com/problems/{title_slug}/"
        })

    with open('leetcode_problems.json', 'w') as f:
        json.dump(problems, f, indent=2)

if __name__ == "__main__":
    scrape_leetcode()