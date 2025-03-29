import { NextResponse } from 'next/server';
import { saveTags, getTags } from '@/lib/db/tags';

// 获取项目的标签树
export async function GET(request, { params }) {
  try {
    const { projectId } = params;

    // 验证项目ID
    if (!projectId) {
      return NextResponse.json({ error: '项目ID不能为空' }, { status: 400 });
    }

    // 获取标签树
    const tags = await getTags(projectId);

    return NextResponse.json({ tags });
  } catch (error) {
    console.error('获取标签树失败:', error);
    return NextResponse.json({ error: error.message || '获取标签树失败' }, { status: 500 });
  }
}

// 更新项目的标签树
export async function PUT(request, { params }) {
  try {
    const { projectId } = params;

    // 验证项目ID
    if (!projectId) {
      return NextResponse.json({ error: '项目ID不能为空' }, { status: 400 });
    }

    // 获取请求体
    const { tags } = await request.json();

    // 验证标签数据
    if (!tags || !Array.isArray(tags)) {
      return NextResponse.json({ error: '标签数据格式不正确' }, { status: 400 });
    }

    // 保存更新后的标签树
    const updatedTags = await saveTags(projectId, tags);

    return NextResponse.json({ tags: updatedTags });
  } catch (error) {
    console.error('更新标签失败:', error);
    return NextResponse.json({ error: error.message || '更新标签失败' }, { status: 500 });
  }
}