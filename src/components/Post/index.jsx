import "./style.less";

const Post = ({username, title, selected}) => {

    return (
        <div className={`post-component ${selected ? "selected" : ""}`}>
            <p>{username}</p>
            <h1>{title}</h1>
        </div>
    );
};

export default Post;