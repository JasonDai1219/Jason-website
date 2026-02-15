import { Icons } from "@/components/common/icons";

interface SocialInterface {
  name: string;
  username: string;
  icon: any;
  link: string;
}

export const SocialLinks: SocialInterface[] = [
  {
    name: "GitHub",
    username: "ruizhedai",
    icon: Icons.gitHub,
    link: "https://github.com/ruizhedai",
  },
  {
    name: "LinkedIn",
    username: "Ruizhe Dai",
    icon: Icons.linkedin,
    link: "https://www.linkedin.com/in/ruizhe-dai/",
  },
  {
    name: "Email",
    username: "ruizhe.dai@seas.upenn.edu",
    icon: Icons.gmail,
    link: "mailto:ruizhe.dai@seas.upenn.edu",
  },
];
