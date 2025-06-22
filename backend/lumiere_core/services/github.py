# In backend/lumiere_core/services/github.py

import os
import re
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List, Any
from github import Github, GithubException, PaginatedList
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
if not GITHUB_TOKEN:
    print("WARNING: GITHUB_ACCESS_TOKEN not found. API calls will be heavily rate-limited.")
    g = Github()
else:
    g = Github(GITHUB_TOKEN)


def _paginated_to_list(paginated_list: PaginatedList, max_items: int = 10) -> List[Dict[str, Any]]:
    items = []
    for i, item in enumerate(paginated_list):
        if i >= max_items:
            break
        item_data = {
            "name": item.name,
            "full_name": item.full_name,
            "description": item.description,
            "html_url": item.html_url,
            "language": item.language,
            "stargazers_count": item.stargazers_count
        }
        items.append(item_data)
    return items


def get_user_profile(username: str) -> Optional[Dict[str, Any]]:
    try:
        user = g.get_user(username)
        return {
            "login": user.login, "name": user.name, "bio": user.bio,
            "html_url": user.html_url, "public_repos": user.public_repos,
            "followers": user.followers, "following": user.following,
        }
    except GithubException:
        return None

def get_user_repos(username: str) -> List[Dict[str, Any]]:
    try:
        user = g.get_user(username)
        return _paginated_to_list(user.get_repos(sort='updated'), max_items=10)
    except GithubException:
        return []

def get_user_starred(username: str) -> List[Dict[str, Any]]:
    try:
        user = g.get_user(username)
        return _paginated_to_list(user.get_starred(), max_items=10)
    except GithubException:
        return []

def get_user_comment_threads(username: str) -> List[Dict[str, Any]]:
    threads = []
    try:
        user = g.get_user(username)
        events = user.get_events()
        # Increase check limit to ensure we find comment events
        max_events_to_check = 50
        comment_events_found = 0
        max_comments_to_process = 5

        for i, event in enumerate(events):
            if i >= max_events_to_check or comment_events_found >= max_comments_to_process:
                break

            if event.type in ['IssueCommentEvent', 'PullRequestReviewCommentEvent']:
                payload = event.payload
                comment_data = payload.get('comment')
                issue_data = payload.get('issue', payload.get('pull_request'))

                if not comment_data or not issue_data or comment_data['user']['login'] != username:
                    continue

                comment_events_found += 1
                repo_name, issue_number = event.repo.name, issue_data['number']

                try:
                    repo_obj = g.get_repo(repo_name)
                    issue_obj = repo_obj.get_issue(number=issue_number)

                    created_at_str = comment_data.get('created_at')
                    if not created_at_str: continue

                    # Correctly parse the ISO 8601 string into a timezone-aware datetime object
                    created_at_dt = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))

                    user_comment = {"id": comment_data['id'], "body": comment_data['body'], "html_url": comment_data['html_url']}

                    replies = []
                    # Fetch comments created *after* the user's comment
                    for reply_comment in issue_obj.get_comments(since=created_at_dt):
                        if reply_comment.user.login != username and reply_comment.id != user_comment['id']:
                            replies.append({"user": reply_comment.user.login, "body": reply_comment.body, "html_url": reply_comment.html_url})

                    threads.append({
                        "repo_name": repo_name, "issue_number": issue_number, "issue_title": issue_data['title'],
                        "issue_url": issue_data['html_url'], "user_comment": user_comment, "replies": replies
                    })
                except GithubException as ge:
                    print(f"Warning: Could not fully process event for {repo_name}#{issue_number}. Skipping. Reason: {ge}")
                    continue
        return threads
    except GithubException as e:
        print(f"GitHub API Error while fetching comment threads: {e}")
        return []

def _parse_github_issue_url(issue_url: str) -> Optional[Tuple[str, str, int]]:
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)/(?:issues|pull)/(\d+)", issue_url)
    if match:
        owner, repo_name, issue_number_str = match.groups()
        return owner, repo_name, int(issue_number_str)
    return None

def scrape_github_issue(issue_url: str) -> Optional[Dict[str, str]]:
    print(f"Fetching GitHub issue via API: {issue_url}")
    parsed_url = _parse_github_issue_url(issue_url)
    if not parsed_url:
        print(f"Error: Could not parse GitHub issue URL: {issue_url}")
        return None
    owner, repo_name, issue_number = parsed_url
    repo_full_name = f"{owner}/{repo_name}"
    try:
        repo = g.get_repo(repo_full_name)
        issue = repo.get_issue(number=issue_number)
        title, description = issue.title, issue.body if issue.body else ""
        full_text_query, repo_url = f"Issue Title: {title}\n\nDescription:\n{description}", f"https://github.com/{owner}/{repo_name}"
        return {"title": title, "description": description, "full_text_query": full_text_query, "repo_url": repo_url}
    except GithubException as e:
        print(f"GitHub API Error: {e.status}, {e.data}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during GitHub API call: {e}")
        return None

def list_open_issues(repo_full_name: str) -> List[Dict[str, Any]]:
    """
    Fetches a list of all open issues for a given repository.
    """
    print(f"Fetching open issues for repository: {repo_full_name}")
    try:
        repo = g.get_repo(repo_full_name)
        open_issues = repo.get_issues(state='open')
        issues_list = []
        for issue in open_issues:
            if not issue.pull_request:
                issues_list.append({
                    "number": issue.number,
                    "title": issue.title,
                    "url": issue.html_url,
                    "author": issue.user.login,
                })
        return issues_list
    except GithubException as e:
        print(f"GitHub API Error while listing issues: {e}")
        return []
