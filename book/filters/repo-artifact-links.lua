-- Keep repository artifacts navigable after Quarto moves a source page from
-- book/chapters/... to book/_output/chapters/.... Relative links that are
-- correct in the checkout otherwise resolve under book/ and break in the
-- published HTML. The artifacts remain in Git; they are not copied into the
-- site bundle (some directories contain large binary research outputs).

local repository = "https://github.com/EigenCharlie/Lending-Club-End-to-End"
local source_ref = "sync/paper-estrella-economic-champion-pipeline-freeze-2026-05-04"
local repository_prefixes = { "data/", "docs/", "models/", "reports/" }

local function strip_parent_segments(target)
  local path = target:gsub("^%./", "")
  while path:sub(1, 3) == "../" do
    path = path:sub(4)
  end
  return path
end

local function is_repository_artifact(path)
  for _, prefix in ipairs(repository_prefixes) do
    if path:sub(1, #prefix) == prefix then
      return true
    end
  end
  return false
end

function Link(link)
  if link.target:match("^[%a][%w+.-]*:") or link.target:sub(1, 1) == "#" then
    return nil
  end

  local path = strip_parent_segments(link.target)
  if not is_repository_artifact(path) then
    return nil
  end

  local path_without_suffix = path:match("^([^?#]+)") or path
  local browser_kind = path_without_suffix:sub(-1) == "/" and "tree" or "blob"
  link.target = repository .. "/" .. browser_kind .. "/" .. source_ref .. "/" .. path
  return link
end
