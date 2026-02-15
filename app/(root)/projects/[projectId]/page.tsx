import Image from "next/image";
import Link from "next/link";
import { redirect } from "next/navigation";

import { Icons } from "@/components/common/icons";
import ProjectDescription from "@/components/projects/project-description";
import { buttonVariants } from "@/components/ui/button";
import ChipContainer from "@/components/ui/chip-container";
import CustomTooltip from "@/components/ui/custom-tooltip";
import { Projects } from "@/config/projects";
import { siteConfig } from "@/config/site";
import { cn, formatDateFromObj } from "@/lib/utils";
import profileImg from "@/public/profile-img.jpg";

interface ProjectPageProps {
  params: Promise<{
    projectId: string;
  }>;
}

export default async function Project({ params }: ProjectPageProps) {
  const { projectId } = await params;
  const project = Projects.find((val) => val.id === projectId);

  if (!project) {
    redirect("/projects");
  }

  const isEarthTimeMachine = project.id === "time-machine-earth";
  const isDsc102Tutor = project.id === "dsc102-ai-tutor";
  const DSC102_YT_EMBED = "https://www.youtube.com/embed/2krRz-4Fx3E";

  return (
    <article className="container relative max-w-3xl py-6 lg:py-10">
      {/* Back button */}
      <Link
        href="/projects"
        className={cn(
          buttonVariants({ variant: "ghost" }),
          "absolute left-[-200px] top-14 hidden xl:inline-flex"
        )}
      >
        <Icons.chevronLeft className="mr-2 h-4 w-4" />
        All Projects
      </Link>

      {/* Header */}
      <div>
        <time className="block text-sm text-muted-foreground">
          {formatDateFromObj(project.startDate)}
        </time>

        <h1 className="flex items-center justify-between mt-2 font-heading text-4xl leading-tight lg:text-5xl">
          {project.companyName}
          <div className="flex items-center">
            {project.githubLink && (
              <CustomTooltip text="Source code">
                <Link href={project.githubLink} target="_blank">
                  <Icons.gitHub className="w-6 ml-4 text-muted-foreground hover:text-foreground" />
                </Link>
              </CustomTooltip>
            )}
            {project.websiteLink && (
              <CustomTooltip text="Demo / External link">
                <Link href={project.websiteLink} target="_blank">
                  <Icons.externalLink className="w-6 ml-4 text-muted-foreground hover:text-foreground" />
                </Link>
              </CustomTooltip>
            )}
          </div>
        </h1>

        <ChipContainer textArr={project.category} />

        {/* Author */}
        <div className="mt-4 flex space-x-4">
          <Link
            href={siteConfig.links.github}
            className="flex items-center space-x-2 text-sm"
          >
            <Image
              src={profileImg}
              alt="author"
              width={42}
              height={42}
              className="rounded-full bg-background"
            />
            <div className="leading-tight">
              <p className="font-medium">{siteConfig.authorName}</p>
              <p className="text-[12px] text-muted-foreground">
                @{siteConfig.username}
              </p>
            </div>
          </Link>
        </div>
      </div>

      {/* ===== Hero Media ===== */}
      {isEarthTimeMachine ? (
        <video
          src="/projects/time-machine-earth/demo.mp4"
          controls
          autoPlay
          loop
          muted
          playsInline
          poster="/projects/time-machine-earth/logo.png"
          className="my-8 w-full rounded-md border bg-muted"
        />
      ) : isDsc102Tutor ? (
        <div className="my-8 aspect-video w-full overflow-hidden rounded-md border bg-muted">
          <iframe
            className="h-full w-full"
            src={DSC102_YT_EMBED}
            title={project.companyName}
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          />
        </div>
      ) : (
        <Image
          src={project.companyLogoImg}
          alt={project.companyName}
          width={720}
          height={405}
          className="my-8 rounded-md border bg-muted"
          priority
        />
      )}

      {/* Tech Stack */}
      <div className="mb-7">
        <h2 className="font-heading text-3xl mb-2">Tech Stack</h2>
        <ChipContainer textArr={project.techStack} />
      </div>

      {/* Description */}
      <div className="mb-7">
        <h2 className="font-heading text-3xl mb-2">Description</h2>
        <ProjectDescription
          paragraphs={project.descriptionDetails.paragraphs}
          bullets={project.descriptionDetails.bullets}
        />
      </div>

      {/* Page Info */}
      <div className="mb-7">
        <h2 className="font-heading text-3xl mb-5">Page Info</h2>
        {project.pagesInfoArr.map((page, idx) => (
          <div key={idx}>
            <h3 className="flex items-center font-heading text-xl mt-3">
              <Icons.star className="h-5 w-5 mr-2" />
              {page.title}
            </h3>
            <p className="mt-1">{page.description}</p>
            {page.imgArr.map((img, i) => (
              <Image
                key={i}
                src={img}
                alt={img}
                width={720}
                height={405}
                className="my-4 rounded-md border bg-muted"
              />
            ))}
          </div>
        ))}
      </div>

      <hr className="mt-12" />

      {/* Footer Back */}
      <div className="flex justify-center py-6 lg:py-10">
        <Link href="/projects" className={cn(buttonVariants({ variant: "ghost" }))}>
          <Icons.chevronLeft className="mr-2 h-4 w-4" />
          All Projects
        </Link>
      </div>
    </article>
  );
}