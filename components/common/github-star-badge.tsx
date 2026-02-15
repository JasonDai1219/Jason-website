"use client";

type GitHubStarBadgeProps = {
  className?: string;
};

// Stars/API removed. Keep a no-op component so existing imports/usages won't break.
export function GitHubStarBadge(_props: GitHubStarBadgeProps) {
  return null;
}

// Some code may import default export with different casing.
export default function GithubStarBadge(_props: GitHubStarBadgeProps) {
  return null;
}
