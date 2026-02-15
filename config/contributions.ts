export interface contributionsInterface {
  repo: string;
  repoOwner: string;
  contibutionDescription: string;
  link?: string; // optional because we handled undefined safely
}

export const contributions: contributionsInterface[] = [];
