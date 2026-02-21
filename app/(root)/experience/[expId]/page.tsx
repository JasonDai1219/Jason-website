import { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { redirect } from "next/navigation";

import { AnimatedSection } from "@/components/common/animated-section";
import { ClientPageWrapper } from "@/components/common/client-page-wrapper";
import { Icons } from "@/components/common/icons";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import ChipContainer from "@/components/ui/chip-container";
import { ResponsiveTabs } from "@/components/ui/responsive-tabs";
import { experiences } from "@/config/experience";
import { siteConfig } from "@/config/site";
import { cn } from "@/lib/utils";

interface ExperienceDetailPageProps {
  params: Promise<{
    expId: string;
  }>;
}

// Helper function to extract year from date
const getYearFromDate = (date: Date): string => {
  return new Date(date).getFullYear().toString();
};

// Helper function to get duration text
const getDurationText = (startDate: Date, endDate: Date | "Present"): string => {
  const startYear = getYearFromDate(startDate);
  const endYear = typeof endDate === "string" ? "Present" : getYearFromDate(endDate);
  return `${startYear} - ${endYear}`;
};

export async function generateMetadata({
  params,
}: ExperienceDetailPageProps): Promise<Metadata> {
  const { expId } = await params;
  const experience = experiences.find((c) => c.id === expId);

  if (!experience) {
    return { title: "Experience Not Found" };
  }

  return {
    title: `${experience.position} at ${experience.company} | Experience`,
    description: `Detailed information about my role as ${experience.position} at ${experience.company}.`,
    alternates: {
      canonical: `${siteConfig.url}/experience/${expId}`,
    },
  };
}

export default async function ExperienceDetailPage({
  params,
}: ExperienceDetailPageProps) {
  const { expId } = await params;
  const experience = experiences.find((c) => c.id === expId);

  if (!experience) {
    redirect("/experience");
  }

  // ✅ 横向 logo 自动放大（OPTIC / HDSI 这类）
  const isWideLogo =
    experience.id === "optic-lab" ||
    experience.id === "hdsi" ||
    experience.company.toLowerCase().includes("optic") ||
    experience.company.toLowerCase().includes("data science institute") ||
    experience.company.toLowerCase().includes("halıcıoglu");

  // ✅ Summary onion parts (safe defaults)
  const summaryHeadline = experience.narrative?.headline ?? "Role Summary";
  const summarySubtitle = experience.narrative?.subtitle;
  const summarySections = experience.narrative?.sections;

  const tabItems = [
    {
      value: "summary",
      label: "Summary",
      content: (
        <AnimatedSection delay={0.3}>
          <div className="space-y-8">
            {/* Onion Layer 1: One-glance overview */}
            <div>
              <h3 className="font-semibold text-sm uppercase tracking-wide text-muted-foreground">
                {summaryHeadline}
              </h3>
              {summarySubtitle && (
                <p className="mt-2 text-sm text-muted-foreground leading-relaxed">
                  {summarySubtitle}
                </p>
              )}
            </div>

            {/* Onion Layer 2: Diagram (visual core) */}
            {experience.diagram && (
              <div>
                <h3 className="font-semibold mb-3 text-sm uppercase tracking-wide text-muted-foreground">
                  System Diagram
                </h3>
                <div className="relative w-full overflow-hidden rounded-lg border bg-muted">
                  <Image
                    src={experience.diagram}
                    alt={`${experience.company} system diagram`}
                    width={1200}
                    height={700}
                    className="w-full h-auto object-contain"
                    priority
                  />
                </div>
              </div>
            )}

            {/* Onion Layer 3: Topic-based narrative (read-along) */}
            {summarySections?.length ? (
              <div className="space-y-6">
                {summarySections.map((sec, i) => (
                  <div key={i} className="space-y-3">
                    <h4 className="text-base font-semibold">{sec.title}</h4>
                    <div className="space-y-2">
                      {sec.body.map((line, j) => (
                        <p key={j} className="text-base leading-relaxed text-foreground/90">
                          {line}
                        </p>
                      ))}
                    </div>
                  </div>
                ))}

                {/* ✅ Evidence block (e.g., CHI PLAY submission screenshot) */}
                {experience.evidence?.image && (
                  <div className="pt-2">
                    <h3 className="font-semibold mb-3 text-sm uppercase tracking-wide text-muted-foreground">
                      Evidence
                    </h3>

                    <figure className="overflow-hidden rounded-lg border bg-muted">
                      <Image
                        src={experience.evidence.image}
                        alt={experience.evidence.alt ?? "Evidence image"}
                        width={1200}
                        height={700}
                        className="w-full h-auto object-contain"
                      />
                      {experience.evidence.caption && (
                        <figcaption className="px-4 py-3 text-sm text-muted-foreground">
                          {experience.evidence.caption}
                        </figcaption>
                      )}
                    </figure>
                  </div>
                )}
              </div>
            ) : (
              // Fallback if narrative not provided
              <div>
                <h3 className="font-semibold mb-4 text-sm uppercase tracking-wide text-muted-foreground">
                  Role Summary
                </h3>
                <ul className="space-y-3">
                  {experience.description.map((desc, idx) => (
                    <li
                      key={idx}
                      className="text-base leading-relaxed flex items-start gap-3"
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                      {desc}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </AnimatedSection>
      ),
    },
    {
      value: "achievements",
      label: "Achievements",
      content: (
        <AnimatedSection delay={0.3}>
          <div>
            <h3 className="font-semibold mb-4 text-sm uppercase tracking-wide text-muted-foreground">
              Key Achievements
            </h3>
            <ul className="space-y-3">
              {experience.achievements.map((achievement, idx) => (
                <li
                  key={idx}
                  className="text-base leading-relaxed flex items-start gap-3"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                  {achievement}
                </li>
              ))}
            </ul>
          </div>
        </AnimatedSection>
      ),
    },
    {
      value: "skills",
      label: "Skills",
      content: (
        <AnimatedSection delay={0.3}>
          <div>
            <h3 className="font-semibold mb-4 text-sm uppercase tracking-wide text-muted-foreground">
              Technologies & Skills
            </h3>
            <ChipContainer textArr={experience.skills} />
          </div>
        </AnimatedSection>
      ),
    },
  ];

  return (
    <ClientPageWrapper>
      <div className="container max-w-4xl mx-auto py-8 px-4">
        <AnimatedSection className="mb-6">
          <Button variant="ghost" size="sm" className="mb-4" asChild>
            <Link href="/experience">
              <Icons.chevronLeft className="mr-2 h-4 w-4" />
              Back to Experience
            </Link>
          </Button>
        </AnimatedSection>

        <AnimatedSection delay={0.2}>
          <Card className="overflow-hidden rounded-lg border bg-background p-2 transition-all duration-300">
            <CardHeader className="pb-6">
              <div className="flex flex-col gap-4">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                  <div className="flex flex-col sm:flex-row items-center sm:items-start gap-4">
                    {experience.logo && (
                      <div
                        className={cn(
                          "rounded-lg border-2 border-border overflow-hidden bg-white flex-shrink-0 flex items-center justify-center",
                          isWideLogo
                            ? "w-40 h-20 sm:w-52 sm:h-24"
                            : "w-16 h-16 sm:w-20 sm:h-20"
                        )}
                      >
                        <Image
                          src={experience.logo}
                          alt={experience.company}
                          width={isWideLogo ? 520 : 96}
                          height={isWideLogo ? 240 : 96}
                          className={cn("w-full h-full object-contain", "p-2")}
                          priority={isWideLogo}
                        />
                      </div>
                    )}

                    <div className="flex-1 text-center sm:text-left">
                      <h1 className="text-xl sm:text-2xl md:text-3xl font-bold mb-2">
                        {experience.position}
                      </h1>
                      <div className="flex items-center justify-center sm:justify-start gap-2 mb-2">
                        <span className="text-md font-medium text-muted-foreground">
                          {experience.company}
                        </span>
                        {experience.companyUrl && (
                          <a
                            href={experience.companyUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-muted-foreground hover:text-foreground transition-colors"
                          >
                            <Icons.externalLink className="w-4 h-4" />
                          </a>
                        )}
                      </div>
                      <p className="text-muted-foreground">{experience.location}</p>
                    </div>
                  </div>

                  <div className="flex justify-center sm:justify-end">
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-primary/10 text-primary border border-primary/20">
                      {getDurationText(experience.startDate, experience.endDate)}
                    </span>
                  </div>
                </div>
              </div>
            </CardHeader>

            <CardContent>
              <ResponsiveTabs items={tabItems} defaultValue="summary" />
            </CardContent>
          </Card>
        </AnimatedSection>

        <AnimatedSection delay={0.4} className="flex justify-center mt-8">
          <Button variant="outline" asChild>
            <Link href="/experience">
              <Icons.chevronLeft className="mr-2 h-4 w-4" />
              View All Experience
            </Link>
          </Button>
        </AnimatedSection>
      </div>
    </ClientPageWrapper>
  );
}