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
    link: "https://github.com/JasonDai1219",
  },
  {
    name: "LinkedIn",
    username: "Ruizhe Dai",
    icon: Icons.linkedin,
    link: "https://www.linkedin.com/in/jason-dai9/",
  },
  {
    name: "Email",
    username: "dai9@seas.upenn.edu",
    icon: Icons.gmail,
    link: "mailto:dai9@seas.upenn.edu",
  },
];
