import "./style.less";

const Post = ({username, title}) => {
    return (
        <div className="post">
            <p>{username}</p>
            <h1>{title}</h1>
        </div>
    );
};

export default Post;